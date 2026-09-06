"""Submission gate (DESIGN.md §5.2, threat rows C2 + C8) — pre-build payload screen.

Runs BEFORE any build or exec, on the raw revealed payload (a set of named
source files). Structural rows enforced here:

- **C8 — weight smuggling:** total source size <= ``MAX_TOTAL_BYTES`` (~1 MB);
  entropy-dense data blobs in source are *flagged* (not rejected) for the §6.3
  manual screen — sub-MB warm-start constants are legal by design.
- **C2 — build-time attack surface:** source-only. Binary content (ELF/PE/
  fatbin/Mach-O, archives, pickle, NUL bytes, non-UTF-8) is rejected, as are
  prebuilt-kernel vehicles: ``.ptx`` files, PTX directives in any file, and
  inline ``asm`` in C/CUDA sources. Detection of inline asm in C is best
  effort by regex — the *guarantee* against what compiles anyway is the
  no-network disposable VM boundary (C2 mitigation), not this gate.
- Python files additionally pass :func:`sn125.sandbox.validate_source`.

Output is a :class:`GateResult` whose ``dq_reason()`` is the string handed to
``RoundFSM.disqualify()`` (miner-fault DQ: fee burned).
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from sn125.sandbox import validate_source

MAX_TOTAL_BYTES = 1_048_576
MAX_FILES = 64
MAX_PATH_LEN = 200

ALLOWED_EXTENSIONS = frozenset([
    ".py", ".c", ".h", ".cu", ".cuh", ".cpp", ".hpp", ".cc",
    ".md", ".txt",
])

_BINARY_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"\x7fELF", "ELF object/shared library"),
    (b"MZ", "PE/DOS executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O binary"),
    (b"\xfe\xed\xfa\xce", "Mach-O binary"),
    (b"\x50\xed\x55\xba", "NVIDIA fatbin"),
    (b"PK\x03\x04", "zip archive"),
    (b"\x1f\x8b", "gzip stream"),
    (b"BZh", "bzip2 stream"),
    (b"\xfd7zXZ\x00", "xz stream"),
    (b"\x28\xb5\x2f\xfd", "zstd stream"),
    (b"7z\xbc\xaf\x27\x1c", "7z archive"),
    (b"Rar!", "rar archive"),
    (b"\x80\x02", "pickle (protocol 2)"),
    (b"\x80\x03", "pickle (protocol 3)"),
    (b"\x80\x04", "pickle (protocol 4)"),
    (b"\x80\x05", "pickle (protocol 5)"),
)

_PTX_DIRECTIVE_RE = re.compile(
    r"^\s*\.(version\s+\d+\.\d+|target\s+sm_\d+|visible\s+\.entry|address_size\s+\d+)",
    re.MULTILINE)

_INLINE_ASM_RE = re.compile(r"(?<![\w.])(__asm__|__asm|asm)\s*(\(|volatile\b|goto\b)")

_C_LIKE_EXTENSIONS = frozenset([".c", ".h", ".cu", ".cuh", ".cpp", ".hpp", ".cc"])

ENTROPY_MIN_BLOB_LEN = 512
ENTROPY_BITS_PER_CHAR = 4.8
DENSE_CONST_TOTAL_BYTES = 65_536
DENSE_NUMERIC_LITERALS = 5_000


def shannon_bits_per_byte(data: bytes) -> float:
    """Shannon entropy of `data` in bits per byte (0.0 for empty)."""
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class GateResult:
    """Outcome of the pre-build gate. ``reasons`` non-empty = reject (DQ);
    ``flags`` are non-fatal entropy notices for the §6.3 manual screen."""
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def dq_reason(self) -> str:
        """Reason string for RoundFSM.disqualify() (miner-fault, fee burned)."""
        if self.ok:
            raise ValueError("submission passed the gate; nothing to DQ")
        return "gate: " + "; ".join(self.reasons)


def _check_path(path: str, out: GateResult) -> bool:
    """Validate a file path; returns False if the file must be skipped."""
    if len(path) > MAX_PATH_LEN:
        out.reasons.append(f"C2: path too long: {path[:40]}...")
        return False
    if path.startswith(("/", "\\")) or ".." in path.split("/") or "\\" in path:
        out.reasons.append(f"C2: non-relative or traversal path: {path}")
        return False
    dot = path.rfind(".")
    ext = path[dot:].lower() if dot >= 0 else ""
    if ext == ".ptx":
        out.reasons.append(f"C2: prebuilt PTX file: {path}")
        return False
    if ext not in ALLOWED_EXTENSIONS:
        out.reasons.append(f"C2: extension not allowed: {path}")
        return False
    return True


def _flag_python_constants(path: str, text: str, out: GateResult) -> None:
    """Entropy screen over Python literals (C8): flag dense data blobs for
    the manual screen. Parse errors are ignored here — validate_source
    already rejects unparseable Python."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    const_bytes = 0
    numeric_literals = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        v = node.value
        if isinstance(v, (int, float, complex)) and not isinstance(v, bool):
            numeric_literals += 1
            continue
        if isinstance(v, (str, bytes)):
            raw = v.encode() if isinstance(v, str) else v
            const_bytes += len(raw)
            if len(raw) >= ENTROPY_MIN_BLOB_LEN:
                ent = shannon_bits_per_byte(raw)
                if ent >= ENTROPY_BITS_PER_CHAR:
                    out.flags.append(
                        f"C8: {path}: {len(raw)}-byte literal at line "
                        f"{node.lineno} with entropy {ent:.2f} bits/byte")
    if const_bytes >= DENSE_CONST_TOTAL_BYTES:
        out.flags.append(
            f"C8: {path}: {const_bytes} total bytes of string/bytes literals")
    if numeric_literals >= DENSE_NUMERIC_LITERALS:
        out.flags.append(
            f"C8: {path}: {numeric_literals} numeric literals (inline data table)")


def validate_submission(files: dict[str, bytes]) -> GateResult:
    """Gate a revealed submission payload: {relative_path: raw_bytes}.

    Rejections accumulate (the miner sees every reason at once); entropy
    flags accumulate separately and never reject on their own.
    """
    out = GateResult()
    if not files:
        out.reasons.append("C2: empty submission")
        return out
    if len(files) > MAX_FILES:
        out.reasons.append(f"C2: too many files: {len(files)} > {MAX_FILES}")
        return out

    total = sum(len(b) for b in files.values())
    if total > MAX_TOTAL_BYTES:
        out.reasons.append(
            f"C8: total source size {total} > {MAX_TOTAL_BYTES} bytes")
        return out

    for path, raw in sorted(files.items()):
        if not _check_path(path, out):
            continue

        magic = next((d for m, d in _BINARY_MAGICS if raw.startswith(m)), None)
        if magic is not None:
            out.reasons.append(f"C2: binary content ({magic}): {path}")
            continue
        if b"\x00" in raw:
            out.reasons.append(f"C2: NUL byte in source: {path}")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            out.reasons.append(f"C2: not valid UTF-8: {path}")
            continue

        if _PTX_DIRECTIVE_RE.search(text):
            out.reasons.append(f"C2: PTX directives in source: {path}")
            continue

        ext = path[path.rfind("."):].lower()
        if ext in _C_LIKE_EXTENSIONS and _INLINE_ASM_RE.search(text):
            out.reasons.append(f"C2: inline asm in C/CUDA source: {path}")
            continue

        if ext == ".py":
            for v in validate_source(text, max_bytes=MAX_TOTAL_BYTES):
                out.reasons.append(f"AST: {path}: {v}")
            _flag_python_constants(path, text, out)

    return out


def validate_single_source(source: str, path: str = "optimizer.py") -> GateResult:
    """Convenience wrapper for the current single-file submission flow."""
    return validate_submission({path: source.encode()})
