"""SN125 — AST validation, sandbox, safe torch/triton proxies, optimizer loading."""
import ast, os, sys, time, logging, re as _re

_log = logging.getLogger("sn125.sandbox")


ALLOWED_IMPORTS = frozenset(["torch", "math", "dataclasses",
                             "collections", "functools", "itertools",
                             "triton", "typing", "enum", "abc"])

_BLOCKED_TYPING_IMPORTS = frozenset(["get_type_hints", "ForwardRef"])

ALLOWED_TRITON_SUBMODULES = frozenset([
    "triton", "triton.language",
])

ALLOWED_TORCH_SUBMODULES = frozenset([
    "torch", "torch.nn", "torch.nn.functional", "torch.nn.init",
    "torch.linalg", "torch.fft", "torch.special", "torch.amp",
    "torch.cuda",
])

FORBIDDEN_NAMES = frozenset([
    "exec", "eval", "compile", "__import__", "importlib", "subprocess",
    "os", "sys", "socket", "http", "urllib", "ctypes", "cffi", "open",
    "file", "breakpoint", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "type", "print", "super",
    "__builtins__",
    "license", "credits", "copyright",
])

_ALLOWED_DUNDER_ATTRS = frozenset([
    "__init__", "__len__", "__iter__", "__next__", "__enter__", "__exit__",
    "__repr__", "__str__", "__bool__", "__call__", "__contains__",
    "__getitem__", "__setitem__", "__delitem__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__",
    "__pow__", "__neg__", "__pos__", "__abs__",
    "__iadd__", "__isub__", "__imul__", "__itruediv__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__and__", "__or__", "__xor__", "__invert__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
    "__matmul__", "__rmatmul__", "__imatmul__",
    "__hash__", "__index__", "__int__", "__float__",
])

_FORBIDDEN_FRAME_ATTRS = frozenset([
    "tb_frame", "tb_next", "tb_lineno", "tb_lasti",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "f_lineno",
    "gi_frame", "gi_code", "gi_yieldfrom",
    "cr_frame", "cr_code", "cr_origin",
    "ag_frame", "ag_code",
    "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars",
    "_os", "_sys", "_thread",
    "inspect", "types", "copy", "keyword", "re", "sys",
    "FunctionType", "CodeType",
    "modules",
    "hub", "multiprocessing", "distributed",
    "_C",
    "storage", "untyped_storage", "storage_type", "storage_offset",
    "_typed_storage",
    "data_ptr", "set_", "as_subclass",
    "record_stream", "resize_", "share_memory_",
    "_share_filename_cpu_", "_share_fd_cpu_", "_share_memory_",
    "_write_file", "_set_from_file",
    "numpy",
    "from_file",
    "register_hook", "register_prehook", "register_post_accumulate_grad_hook",
    "register_backward_hook", "register_forward_hook", "register_forward_pre_hook",
    "register_full_backward_hook", "register_full_backward_pre_hook",
    "get_type_hints", "ForwardRef", "_evaluate",
    "inline_asm_elementwise", "extern_elementwise",
    "device_print", "device_assert",
    "module_from_spec", "module_finder",
    "format", "format_map",
])


class SandboxViolation(Exception):
    pass



_TAMPER_CLASS_NAMES = ("torch.Tensor", "torch.nn.Module", "torch.nn.Linear",
                       "torch.nn.LayerNorm", "torch.nn.Embedding", "torch.nn.Parameter")

_TAMPER_MODULE_NAMES = ("torch.nn.functional", "torch.nn.utils")
_TAMPER_TORCH_FUNCS = frozenset({
    "isfinite", "as_tensor", "zeros_like", "no_grad", "inference_mode",
    "enable_grad", "clamp", "norm", "stack", "cat",
})

_SENTINEL = object()


def _resolve(dotted):
    import torch
    obj = torch
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def _stable_getattr_attrs(obj, names):
    """Snapshot {name: object} for `names` on `obj`, keeping only entries whose
    getattr identity is stable (skips implicit classmethods like
    __init_subclass__ / __torch_function__ that rebuild a bound object per
    access, which would otherwise false-trip)."""
    out = {}
    for name in names:
        try:
            a = getattr(obj, name)
            b = getattr(obj, name)
        except Exception:
            continue
        if a is b:
            out[name] = a
    return out


def snapshot_torch_identities() -> dict:
    """Capture the harness-trusted torch surface for tamper detection. Must be
    called while torch is clean (before any miner code runs in this process)."""
    try:
        from torch._dynamo.mutation_guard import install_generation_tagging_init
        install_generation_tagging_init()
    except Exception:
        pass
    snap = {"_classes": {}, "_modules": {}}
    for cname in _TAMPER_CLASS_NAMES:
        try:
            cls = _resolve(cname)
        except Exception:
            continue
        snap["_classes"][cname] = _stable_getattr_attrs(cls, dir(cls))
    for mname in _TAMPER_MODULE_NAMES:
        try:
            mod = _resolve(mname)
        except Exception:
            continue
        snap["_modules"][mname] = _stable_getattr_attrs(mod, dir(mod))
    try:
        import torch
        snap["_modules"]["torch"] = _stable_getattr_attrs(torch, _TAMPER_TORCH_FUNCS)
    except Exception:
        pass
    return snap


_PRISTINE_TORCH = None


def _ensure_pristine():
    global _PRISTINE_TORCH
    if _PRISTINE_TORCH is None:
        try:
            _PRISTINE_TORCH = snapshot_torch_identities()
        except Exception:
            _PRISTINE_TORCH = {}
    return _PRISTINE_TORCH


def check_torch_tamper(restore: bool = True, max_report: int = 12) -> list[str]:
    """Return a list of `obj.attr` entries the optimizer monkey-patched on the
    trusted torch surface (replaced existing methods, or shadowed inherited ones
    on the trusted classes). If `restore` is True, each is reset to its pristine
    object so the process is clean for the next submission. Empty == untampered."""
    pristine = _ensure_pristine()
    if not pristine:
        return []
    bad: list[str] = []

    for cname, attrs in pristine.get("_classes", {}).items():
        try:
            cls = _resolve(cname)
        except Exception:
            continue
        for name, original in attrs.items():
            try:
                current = getattr(cls, name)
            except Exception:
                current = _SENTINEL
            if current is original:
                continue
            bad.append(f"{cname}.{name}")
            if restore:
                try:
                    if name in vars(cls):
                        if original is getattr(super(cls, cls), name, _SENTINEL):
                            delattr(cls, name)
                        else:
                            setattr(cls, name, original)
                except Exception:
                    pass

    for mname, attrs in pristine.get("_modules", {}).items():
        try:
            mod = _resolve(mname)
        except Exception:
            continue
        for name, original in attrs.items():
            try:
                current = getattr(mod, name)
            except Exception:
                current = _SENTINEL
            if current is original:
                continue
            bad.append(f"{mname}.{name}")
            if restore:
                try:
                    setattr(mod, name, original)
                except Exception:
                    pass

    return bad[:max_report] if len(bad) > max_report else bad


def rebaseline_torch_identities() -> None:
    """Re-capture the trusted torch surface as the NEW pristine baseline.

    DIAGNOSTIC USE ONLY — for the harness-controlled ``torch.compile`` path.
    TorchDynamo performs a one-time global monkeypatch of
    ``nn.Module.__init__``/``__setstate__`` the first time it compiles a module.
    That swap changes the identity of trusted methods and would otherwise
    false-trip :func:`check_torch_tamper` on the first optimizer step.

    Re-snapshotting here folds Dynamo's *legitimate* patches into the trusted
    baseline. This does NOT weaken the scoring-integrity guarantee BECAUSE compile
    is invoked by the harness (a fixed ``--compile`` diagnostic flag, never miner
    input) and the miner optimizer executes in a SEPARATE process over CUDA IPC —
    it has not run, and cannot run, before this call. CALLER CONTRACT: invoke only
    while torch is clean of miner influence, i.e. immediately after a harness-only
    warm-up forward/backward that triggers compilation and BEFORE the scoring loop
    runs any ``opt.step()``. Any tampering a miner attempts during the scored steps
    is still caught against this refreshed baseline."""
    global _PRISTINE_TORCH
    _PRISTINE_TORCH = snapshot_torch_identities()


def _attr_assignment_root(target: "ast.AST"):
    """For an assignment/deletion target that is an attribute access (`a.b.c`),
    return the root Name id (`a`) and the attribute being set (`c`). Returns
    (None, None) if the target is not an attribute access."""
    if not isinstance(target, ast.Attribute):
        return None, None
    attr = target.attr
    node = target
    while isinstance(node, ast.Attribute):
        node = node.value
    root = node.id if isinstance(node, ast.Name) else None
    return root, attr


_FORMAT_DUNDER_RE = _re.compile(r'__\w+__')

def validate_source(source: str, max_bytes: int = 1_048_576, allow_native: bool = False) -> list[str]:
    """Return list of violations. Empty = safe."""
    violations = []
    if len(source.encode()) > max_bytes:
        violations.append(f"Source too large: {len(source.encode())} > {max_bytes}")
        return violations
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        violations.append(f"Syntax error: {e}")
        return violations

    _allowed = ALLOWED_IMPORTS | ({"ctypes"} if allow_native else set())
    _forbidden_names = FORBIDDEN_NAMES - ({"ctypes"} if allow_native else set())

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            _s = node.value.decode() if isinstance(node.value, bytes) else node.value
            for m in _FORMAT_DUNDER_RE.finditer(_s):
                dunder = m.group()
                if dunder not in _ALLOWED_DUNDER_ATTRS:
                    violations.append(f"Forbidden dunder in string/bytes literal: {dunder}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _allowed:
                    violations.append(f"Forbidden import: {alias.name}")
                if alias.name.startswith("torch.") and alias.name not in ALLOWED_TORCH_SUBMODULES:
                    violations.append(f"Forbidden torch submodule: {alias.name}")
                if alias.name.startswith("triton.") and alias.name not in ALLOWED_TRITON_SUBMODULES:
                    violations.append(f"Forbidden triton submodule: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or (hasattr(node, 'level') and node.level > 0):
                violations.append("Relative imports forbidden")
            if node.module:
                root = node.module.split(".")[0]
                if root not in _allowed:
                    violations.append(f"Forbidden import from: {node.module}")
                if node.module.startswith("torch.") and node.module not in ALLOWED_TORCH_SUBMODULES:
                    violations.append(f"Forbidden torch submodule: {node.module}")
                if node.module.startswith("triton.") and node.module not in ALLOWED_TRITON_SUBMODULES:
                    violations.append(f"Forbidden triton submodule: {node.module}")
                if node.module == "torch" and node.names:
                    _SAFE_TORCH_ATTRS = {"Tensor", "dtype", "device", "Size", "no_grad",
                                         "inference_mode", "enable_grad", "nn",
                                         "amp", "linalg", "fft", "special",
                                         "manual_seed", "Generator", "seed",
                                         "initial_seed", "get_rng_state", "set_rng_state"}
                    for alias in node.names:
                        sub = f"torch.{alias.name}"
                        if alias.name not in _SAFE_TORCH_ATTRS and sub not in ALLOWED_TORCH_SUBMODULES:
                            violations.append(f"Forbidden torch import: {alias.name}")
                if node.module == "triton" and node.names:
                    _SAFE_TRITON_ATTRS = {"jit", "cdiv", "language"}
                    for alias in node.names:
                        sub = f"triton.{alias.name}"
                        if alias.name not in _SAFE_TRITON_ATTRS and sub not in ALLOWED_TRITON_SUBMODULES:
                            violations.append(f"Forbidden triton import: {alias.name}")
                _BLOCKED_TL_IMPORTS = {"inline_asm_elementwise", "extern_elementwise"}
                if node.module == "triton.language" and node.names:
                    for alias in node.names:
                        if alias.name in _BLOCKED_TL_IMPORTS:
                            violations.append(f"Forbidden triton.language import: {alias.name}")
                if node.module == "typing" and node.names:
                    for alias in node.names:
                        if alias.name in _BLOCKED_TYPING_IMPORTS:
                            violations.append(f"Forbidden typing import: {alias.name}")
        elif isinstance(node, ast.Name) and node.id in _forbidden_names:
            violations.append(f"Forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") and a.endswith("__") and a not in _ALLOWED_DUNDER_ATTRS:
                violations.append(f"Forbidden dunder attribute: {a}")
            elif a in _FORBIDDEN_FRAME_ATTRS:
                violations.append(f"Forbidden attribute: {a}")
        elif isinstance(node, ast.MatchClass) if hasattr(ast, "MatchClass") else False:
            for kwd in node.kwd_attrs:
                if kwd.startswith("__") and kwd.endswith("__") and kwd not in _ALLOWED_DUNDER_ATTRS:
                    violations.append(f"Forbidden dunder in match pattern: {kwd}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name.endswith("__") and node.name not in _ALLOWED_DUNDER_ATTRS:
                violations.append(f"Forbidden dunder method definition: {node.name}")
            if isinstance(node, ast.AsyncFunctionDef):
                violations.append(f"Async functions forbidden: {node.name}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            _targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete))
                        else [node.target])
            for _t in _targets:
                _root, _attr = _attr_assignment_root(_t)
                if _root is not None and _root not in ("self", "cls"):
                    violations.append(
                        f"Forbidden attribute assignment: {_root}.{_attr} "
                        f"(only `self.*` may be assigned)")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile", "__import__"):
                violations.append(f"Forbidden call: {node.func.id}")
        elif hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):
            violations.append("Exception groups (except*) forbidden")
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                violations.append("Bare 'except:' forbidden (use 'except Exception:')")
            else:
                _FORBIDDEN_EXCEPT = {"BaseException", "GeneratorExit", "KeyboardInterrupt", "SystemExit"}
                names = []
                if isinstance(node.type, ast.Name):
                    names = [node.type.id]
                elif isinstance(node.type, ast.Tuple):
                    names = [e.id for e in node.type.elts if isinstance(e, ast.Name)]
                for eid in names:
                    if eid in _FORBIDDEN_EXCEPT:
                        violations.append(f"'except {eid}' forbidden (use 'except Exception:')")
    return violations


def _make_safe_torch():
    """Create a restricted torch module proxy that blocks dangerous operations."""
    import types
    import torch

    _BLOCKED_ATTRS = frozenset(["torch", "save", "load", "compile", "package",
                                 "onnx", "jit", "hub", "utils", "multiprocessing",
                                 "distributed", "_C", "profiler", "serialization",
                                 "PyTorchFileReader", "PyTorchFileWriter",
                                 "UntypedStorage", "StorageBase", "TypedStorage",
                                 "BFloat16Storage", "BoolStorage", "ByteStorage",
                                 "CharStorage", "ComplexDoubleStorage", "ComplexFloatStorage",
                                 "DoubleStorage", "FloatStorage", "HalfStorage",
                                 "IntStorage", "LongStorage", "ShortStorage",
                                 "QInt8Storage", "QInt32Storage", "QUInt8Storage",
                                 "QUInt2x4Storage", "QUInt4x2Storage", "Storage",
                                 "ScriptModule", "ScriptFunction", "ScriptMethod",
                                 "ScriptObject", "ScriptClass", "ScriptClassFunction",
                                 "ScriptDict", "ScriptList", "ScriptDictIterator",
                                 "ScriptDictKeyIterator", "ScriptListIterator",
                                 "ScriptObjectProperty", "ScriptModuleSerializer",
                                 "CompilationUnit", "StaticModule", "LiteScriptModule",
                                 "TracingState", "Graph", "Node", "Block", "Use", "Value",
                                 "DeserializationStorageContext", "SerializationStorageContext",
                                 "DeepCopyMemoTable",
                                 "from_file", "get_file_path", "FileCheck",
                                 "read_vitals", "set_vital", "vitals_enabled",
                                 "fork", "prepare_multiprocessing_environment",
                                 "import_ir_module", "import_ir_module_from_buffer",
                                 "BenchmarkExecutionStats", "ExecutionPlan", "GraphExecutorState",
                                 "set_num_threads", "set_num_interop_threads", "init_num_threads",
                                 "is_storage",
                                 "ThroughputBenchmark", "BenchmarkConfig",
                                 "Future", "FutureType",
                                 "ConcreteModuleTypeBuilder", "ConcreteModuleType",
                                 "Code", "Gradient", "AliasDb", "ErrorReport",
                                 "CallStack", "IODescriptor", "Capsule",
                                 "LockingLogger", "LoggerBase", "NoopLogger",
                                 "ModuleDict", "ParameterDict", "BufferDict",
                                 "AggregationType",
                                 "parse_ir", "parse_schema", "parse_type_comment",
                                 "merge_type_from_type_comment", "unify_type_list",
                                 "Argument", "ArgumentSpec", "CompleteArgumentSpec",
                                 "FunctionSchema", "OperatorInfo", "InferredType",
                                 "NumberType", "PyObjectType", "RRefType", "AwaitType",
                                 "JITException", "FatalError",
                                 "get_device_module",
                                 "ClassType", "Type", "AnyType", "BoolType", "ComplexType",
                                 "DeviceObjType", "DictType", "EnumType", "FloatType",
                                 "IntType", "InterfaceType", "ListType", "NoneType",
                                 "OptionalType", "StreamObjType", "StringType",
                                 "SymBoolType", "SymIntType", "TensorType", "TupleType",
                                 "UnionType",
                                 "DispatchKey", "DispatchKeySet", "ExcludeDispatchKeyGuard",
                                 "Tag", "PRIVATE_OPS", "DisableTorchFunctionSubclass",
                                 "manual_seed", "seed", "set_rng_state",
                                 "set_default_device", "set_default_dtype",
                                 "set_default_tensor_type",
                                 "set_float32_matmul_precision",
                                 "set_deterministic_debug_mode",
                                 "use_deterministic_algorithms",
                                 "set_anomaly_enabled",
                                 "set_flush_denormal",
                                 "set_warn_always", "set_printoptions",
                                 "set_autocast_enabled", "set_autocast_dtype",
                                 "set_autocast_gpu_dtype", "set_autocast_cpu_enabled",
                                 "set_autocast_cpu_dtype", "set_autocast_ipu_enabled",
                                 "set_autocast_ipu_dtype", "set_autocast_xla_enabled",
                                 "set_autocast_xla_dtype", "set_autocast_cache_enabled",
                                 "DataParallel",
                                 "autocast", "autocast_increment_nesting",
                                 "autocast_decrement_nesting", "clear_autocast_cache",
                                 "GradScaler",
                                 "Event", "Stream",
                                 "DisableTorchFunction",
                                 ])

    def _proxy_module(real_mod, name):
        proxy = types.ModuleType(name)
        for attr in dir(real_mod):
            if attr.startswith("_") or attr in _BLOCKED_ATTRS:
                continue
            try:
                val = getattr(real_mod, attr)
                if isinstance(val, types.ModuleType):
                    continue
                setattr(proxy, attr, val)
            except Exception:
                pass
        return proxy

    safe = types.ModuleType("torch")
    for attr in dir(torch):
        if attr.startswith("_") or attr in _BLOCKED_ATTRS:
            continue
        try:
            val = getattr(torch, attr)
            if isinstance(val, types.ModuleType):
                continue
            setattr(safe, attr, val)
        except Exception:
            pass
    _FOREACH_ALLOWED = (
        "_foreach_add", "_foreach_add_", "_foreach_sub", "_foreach_sub_",
        "_foreach_mul", "_foreach_mul_", "_foreach_div", "_foreach_div_",
        "_foreach_sqrt", "_foreach_sqrt_", "_foreach_addcmul_",
        "_foreach_addcdiv_", "_foreach_copy_",
    )
    for attr in _FOREACH_ALLOWED:
        if hasattr(torch, attr):
            setattr(safe, attr, getattr(torch, attr))

    safe.nn = _proxy_module(torch.nn, "torch.nn")
    safe.nn.functional = _proxy_module(torch.nn.functional, "torch.nn.functional")
    safe.nn.init = _proxy_module(torch.nn.init, "torch.nn.init")
    for cls_name in ("Module", "Parameter", "Linear", "LayerNorm", "Embedding"):
        if hasattr(torch.nn, cls_name):
            setattr(safe.nn, cls_name, getattr(torch.nn, cls_name))
    safe.linalg = _proxy_module(torch.linalg, "torch.linalg")
    safe.fft = _proxy_module(torch.fft, "torch.fft")
    safe.special = _proxy_module(torch.special, "torch.special")
    safe.amp = _proxy_module(torch.amp, "torch.amp")
    safe.cuda = _proxy_module(torch.cuda, "torch.cuda")
    _CUDA_BLOCKED = {"set_per_process_memory_fraction", "mem_get_info",
                     "reset_peak_memory_stats", "memory_stats", "set_device",
                     "init",
                     "manual_seed", "manual_seed_all", "seed", "seed_all",
                     "set_rng_state", "set_rng_state_all",
                     "change_current_allocator", "CUDAPluggableAllocator",
                     "caching_allocator_alloc", "caching_allocator_delete",
                     "caching_allocator_enable", "empty_cache",
                     "CUDAGraph", "graph", "graph_pool_handle",
                     "make_graphed_callables", "is_current_stream_capturing",
                     "set_stream", "ExternalStream", "StreamContext", "stream",
                     "Event",
                     "MemPool", "MemPoolContext", "use_mem_pool",
                     "ipc_collect",
                     "set_sync_debug_mode",
                     "reset_accumulated_memory_stats", "reset_max_memory_allocated",
                     "reset_max_memory_cached",
                     }
    for fn in _CUDA_BLOCKED:
        if hasattr(safe.cuda, fn):
            delattr(safe.cuda, fn)
    safe.Tensor = torch.Tensor
    safe.dtype = torch.dtype
    return safe


def _make_safe_triton():
    """Create a restricted triton module proxy."""
    import types
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    safe = types.ModuleType("triton")
    safe.jit = triton.jit
    if hasattr(triton, "cdiv"):
        safe.cdiv = triton.cdiv

    _ALLOWED_TL = frozenset([
        "bfloat16", "float16", "float32", "float64", "int1", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "pi32_t", "void",
        "float8e4b15", "float8e4b8", "float8e4nv", "float8e5", "float8e5b16",
        "dtype", "block_type", "pointer_type", "function_type", "nv_tma_desc_type",
        "const", "constexpr",
        "load", "store", "make_block_ptr", "advance",
        "abs", "add", "ceil", "clamp", "cos", "div_rn", "dot", "dot_scaled", "erf",
        "exp", "exp2", "fdiv", "flip", "floor", "fma", "log", "log2", "maximum", "minimum",
        "rsqrt", "sigmoid", "sin", "softmax", "sqrt", "sqrt_rn", "umulhi",
        "argmax", "argmin", "max", "min", "reduce", "sum", "xor_sum",
        "cumsum", "cumprod", "associative_scan", "histogram",
        "arange", "broadcast", "broadcast_to", "cat", "expand_dims", "full",
        "interleave", "join", "ravel", "reshape", "split", "trans", "view",
        "where", "zeros", "zeros_like", "permute", "sort", "swizzle2d", "cast",
        "program_id", "num_programs", "range", "static_range", "static_assert",
        "static_print", "debug_barrier", "assume",
        "max_contiguous", "max_constancy", "multiple_of",
        "philox", "philox_impl", "rand", "rand4x", "randint", "randint4x",
        "randn", "randn4x", "pair_uniform_to_normal", "uint_to_uniform_float",
        "atomic_add", "atomic_and", "atomic_cas", "atomic_max", "atomic_min",
        "atomic_or", "atomic_xchg", "atomic_xor",
        "PropagateNan", "TRITON_MAX_TENSOR_NUMEL",
        "cdiv",
    ])
    safe_tl = types.ModuleType("triton.language")
    for attr in _ALLOWED_TL:
        val = getattr(tl, attr, None)
        if val is not None:
            setattr(safe_tl, attr, val)
    safe.language = safe_tl
    return safe


def load_optimizer_sandboxed(source: str, build_dir: str = "") -> type:
    """Compile and load Optimizer class from source in restricted namespace."""
    import hashlib as _hl
    import linecache
    import torch

    has_native = bool(build_dir)
    violations = validate_source(source, allow_native=has_native)
    if violations:
        raise SandboxViolation(f"Source failed validation: {violations}")

    allowed_modules = {"torch": _make_safe_torch()}
    safe_triton = _make_safe_triton()
    if safe_triton is not None:
        allowed_modules["triton"] = safe_triton
    for name in ALLOWED_IMPORTS:
        if name in ("torch", "triton"):
            continue
        try:
            import importlib
            allowed_modules[name] = importlib.import_module(name)
        except ImportError:
            pass

    raw = __builtins__ if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k) for k in dir(__builtins__)
    }
    restricted_builtins = dict(raw)
    for name in ["exec", "eval", "compile", "open", "breakpoint",
                  "globals", "locals", "vars", "dir",
                  "getattr", "setattr", "delattr", "hasattr",
                  "input", "help", "print", "super",
                  "type",
                  "license", "credits", "copyright",
                  ]:
        restricted_builtins.pop(name, None)

    safe_torch = allowed_modules["torch"]
    _safe_submodules = {
        "torch.nn": safe_torch.nn,
        "torch.nn.functional": safe_torch.nn.functional,
        "torch.nn.init": safe_torch.nn.init,
        "torch.linalg": safe_torch.linalg,
        "torch.fft": safe_torch.fft,
        "torch.special": safe_torch.special,
        "torch.amp": safe_torch.amp,
        "torch.cuda": safe_torch.cuda,
    }
    _safe_triton = allowed_modules.get("triton")
    if _safe_triton is not None:
        _safe_submodules["triton.language"] = _safe_triton.language
    real_import = __import__
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        _extra_allowed = {"ctypes"} if build_dir else set()
        if root not in ALLOWED_IMPORTS and root not in _extra_allowed:
            raise ImportError(f"Import of '{name}' is not allowed")
        if root == "torch":
            if name == "torch":
                return safe_torch
            if name in _safe_submodules:
                if fromlist:
                    return _safe_submodules[name]
                return safe_torch
            raise ImportError(f"Import of '{name}' is not allowed (torch submodule not whitelisted)")
        if root == "triton":
            if _safe_triton is None:
                raise ImportError("triton is not available")
            if name == "triton":
                return _safe_triton
            if name in _safe_submodules:
                if fromlist:
                    return _safe_submodules[name]
                return _safe_triton
            raise ImportError(f"Import of '{name}' is not allowed (triton submodule not whitelisted)")
        if root == "ctypes" and "ctypes" in allowed_modules:
            return allowed_modules["ctypes"]
        return real_import(name, globals, locals, fromlist, level)
    restricted_builtins["__import__"] = safe_import

    namespace = {"__builtins__": restricted_builtins}
    namespace.update(allowed_modules)

    if build_dir and os.path.isdir(build_dir):
        import ctypes as _real_ctypes
        import types as _types
        _safe_ctypes = _types.ModuleType("ctypes")
        _safe_ctypes.c_int = _real_ctypes.c_int
        _safe_ctypes.c_float = _real_ctypes.c_float
        _safe_ctypes.c_double = _real_ctypes.c_double
        _safe_ctypes.c_void_p = _real_ctypes.c_void_p
        _safe_ctypes.c_char_p = _real_ctypes.c_char_p
        _safe_ctypes.c_size_t = _real_ctypes.c_size_t
        _safe_ctypes.c_int64 = _real_ctypes.c_int64
        _safe_ctypes.c_uint64 = _real_ctypes.c_uint64
        _safe_ctypes.POINTER = _real_ctypes.POINTER
        _safe_ctypes.byref = _real_ctypes.byref
        _safe_ctypes.cast = _real_ctypes.cast
        _build = os.path.realpath(build_dir)
        class _SafeCDLL:
            __slots__ = ()
            _BLOCKED = frozenset({"_handle", "_name", "_FuncPtr"})
            _libs = {}
            def __init__(self, path, *a, **kw):
                real = os.path.realpath(path)
                if not (real.startswith(_build + os.sep) or real == _build):
                    raise PermissionError(f"Can only load .so from {_build}")
                if not real.endswith(".so"):
                    raise PermissionError("Can only load .so files")
                _SafeCDLL._libs[id(self)] = _real_ctypes.CDLL(real, *a, **kw)
            def __del__(self):
                _SafeCDLL._libs.pop(id(self), None)
            def __getattribute__(self, name):
                if name in ("__repr__", "__str__"):
                    return object.__getattribute__(self, name)
                if name.startswith("_"):
                    raise AttributeError(f"Access denied: {name}")
                lib = _SafeCDLL._libs.get(id(self))
                if lib is None:
                    raise AttributeError("CDLL not initialized")
                attr = getattr(lib, name)
                if not callable(attr):
                    raise AttributeError(f"Only callable symbols exposed: {name}")
                return attr
        _safe_ctypes.CDLL = _SafeCDLL
        allowed_modules["ctypes"] = _safe_ctypes
        namespace["ctypes"] = _safe_ctypes

    _src_hash = _hl.sha256(source.encode()).hexdigest()[:16]
    _tmppath = f"/tmp/_sn125_opt_{_src_hash}_{os.getpid()}_{time.time_ns()}.py"
    if _tmppath not in linecache.cache:
        try:
            _fd = os.open(_tmppath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(_fd, source.encode())
            finally:
                os.close(_fd)
        except FileExistsError:
            pass
        linecache.cache[_tmppath] = (len(source), None, source.splitlines(True), _tmppath)
    _ensure_pristine()
    code = compile(source, _tmppath, "exec")
    exec(code, namespace)  # noqa: S102 — intentional sandboxed exec

    _tampered = check_torch_tamper(restore=True)
    if _tampered:
        raise SandboxViolation(
            f"Source tampered with trusted torch global state at import: {_tampered}")

    if "Optimizer" not in namespace:
        raise SandboxViolation("Source must define a class named 'Optimizer'")
    return namespace["Optimizer"]
