"""Process-level miner sandbox for CONTAINER hosts (RunPod pods).

The namespace sandbox in ``training.py`` (``unshare --mount --net --ipc --uts
--pid`` + bind mounts + priv drop) and the box-level egress lockdown in
``cloud.py`` (iptables OUTPUT DROP, probed dead) both need capabilities a
Docker container does not have: CAP_SYS_ADMIN for namespaces and mounts,
CAP_NET_ADMIN for netfilter. A RunPod pod (probed 2026-09-04, Secure Cloud,
``runpod/pytorch`` image) runs as root with only the default Docker cap set,
the default Docker seccomp profile (every ``unshare`` flavour is EPERM,
user namespaces included), AppArmor ``docker-default`` and cgroup v1.

What a container DOES allow is enough for a process-level boundary:

* ``setuid``/``setgid`` to ``nobody`` (65534) — the worker keeps its
  privilege drop;
* ``PR_SET_NO_NEW_PRIVS`` — no way back up via setuid binaries;
* stacking a **seccomp-BPF filter** on top of Docker's — the kernel then
  refuses, for the worker and every process it forks or execs, the syscalls
  that would open a network path or break the process boundary.

This module builds and installs that filter with ``ctypes`` only (no
libseccomp), then PROVES it took effect with a positive probe (an
``AF_INET`` socket must fail with EPERM while ``AF_UNIX`` still works — the
CUDA-IPC optimizer process and ``torch.multiprocessing`` need UNIX sockets).
The worker reports the probe result inside its nonce-authenticated output,
so the parent's ``_enforce_sandbox_attestation`` can fail closed on it
exactly as it does on the namespace attestation.

Threat model delta vs the namespace sandbox (documented, not hidden):

* No private ``/tmp``/mount view: the worker gets a fresh per-run directory
  (mode 0700, owned by nobody) for build products and compiler caches, and
  the trusted clean-score worker never reads a cache the miner could have
  written (``score_worker._worker_env`` pins fresh, root-owned cache dirs).
  Root-owned files stay unwritable to nobody by ordinary DAC.
* No PID namespace: ``ptrace``, ``process_vm_readv/writev``, ``pidfd_getfd``
  and ``kcmp`` are denied by the filter and Yama ``ptrace_scope=1`` +
  uid separation already refuse cross-uid inspection.
* No cgroup v2 RSS cap on the pod: host memory is bounded by
  ``RLIMIT_DATA`` instead (anonymous mappings), on top of ``RLIMIT_AS``.
* Egress is denied per PROCESS TREE, not per box. The validator's own
  processes on the box (shard staging, checkpoint upload) keep the network.
  Miner code cannot reach them: it is uid 65534, they are root.

The filter is x86_64-only and fails closed elsewhere (raises before any
miner code runs).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import platform
import socket
import struct

SECCOMP_MODE_FILTER = 2
SECCOMP_SET_MODE_FILTER = 1
SECCOMP_FILTER_FLAG_TSYNC = 1
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
AUDIT_ARCH_X86_64 = 0xC000003E
X32_SYSCALL_BIT = 0x40000000
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
NR_SECCOMP = 317

BPF_LD, BPF_W, BPF_ABS = 0x00, 0x00, 0x20
BPF_JMP, BPF_JEQ, BPF_JGE, BPF_K = 0x05, 0x10, 0x30, 0x00
BPF_RET = 0x06

_OFF_NR = 0
_OFF_ARCH = 4
_OFF_ARG0_LO = 16

NR = {
    "socket": 41, "socketpair": 53, "ptrace": 101,
    "pivot_root": 155, "chroot": 161, "acct": 163, "settimeofday": 164,
    "mount": 165, "umount2": 166, "swapon": 167, "swapoff": 168, "reboot": 169,
    "init_module": 175, "delete_module": 176, "clock_settime": 227,
    "kexec_load": 246, "add_key": 248, "request_key": 249, "keyctl": 250,
    "unshare": 272, "perf_event_open": 298, "open_by_handle_at": 304,
    "setns": 308, "process_vm_readv": 310, "process_vm_writev": 311,
    "kcmp": 312, "finit_module": 313, "kexec_file_load": 320, "bpf": 321,
    "userfaultfd": 323, "io_uring_setup": 425, "io_uring_enter": 426,
    "io_uring_register": 427, "pidfd_getfd": 438,
    "mount_setattr": 442, "fsopen": 430, "fsmount": 432, "open_tree": 428,
    "move_mount": 429,
}

DENIED = (
    "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd", "kcmp",
    "io_uring_setup", "io_uring_enter", "io_uring_register",
    "unshare", "setns", "mount", "umount2", "pivot_root", "chroot",
    "mount_setattr", "fsopen", "fsmount", "open_tree", "move_mount",
    "open_by_handle_at",
    "bpf", "perf_event_open", "userfaultfd",
    "add_key", "request_key", "keyctl",
    "init_module", "finit_module", "delete_module",
    "kexec_load", "kexec_file_load", "reboot", "swapon", "swapoff", "acct",
    "settimeofday", "clock_settime",
)

AF_UNIX = 1

SANDBOX_MODE_NAMESPACES = "namespaces"
SANDBOX_MODE_SECCOMP = "seccomp"


def _stmt(code: int, k: int) -> bytes:
    return struct.pack("HBBI", code, 0, 0, k & 0xFFFFFFFF)


def _jump(code: int, k: int, jt: int, jf: int) -> bytes:
    return struct.pack("HBBI", code, jt, jf, k & 0xFFFFFFFF)


def build_filter(denied: tuple[str, ...] = DENIED,
                 allowed_socket_domains: tuple[int, ...] = (AF_UNIX,)) -> bytes:
    """Return the packed ``struct sock_filter[]`` for the miner filter.

    Layout (every jump is forward, offsets are computed from the tail):

        ld arch; jne X86_64 -> KILL
        ld nr;   jge X32_BIT -> KILL
        jeq socket -> [ld arg0; jeq AF_UNIX -> ALLOW; ... ; -> EPERM]
        jeq denied_i -> EPERM   (for every denied syscall)
        ALLOW
    """
    nrs = sorted({NR[name] for name in denied})
    ret_allow = _stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
    ret_eperm = _stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | errno.EPERM)
    ret_kill = _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS)

    sock_block = [_stmt(BPF_LD | BPF_W | BPF_ABS, _OFF_ARG0_LO)]
    n_dom = len(allowed_socket_domains)
    for i, dom in enumerate(allowed_socket_domains):
        remaining = n_dom - i - 1
        sock_block.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, dom, remaining + 1, 0))
    sock_block.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | errno.EPERM))
    sock_block.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW))

    n_dispatch = 1 + len(nrs)
    head: list[bytes] = []
    head.append(_stmt(BPF_LD | BPF_W | BPF_ABS, _OFF_ARCH))
    idx_ld_nr = 2
    idx_x32 = 3
    idx_dispatch0 = 4
    idx_allow = idx_dispatch0 + n_dispatch
    idx_eperm = idx_allow + 1
    idx_kill = idx_allow + 2
    idx_sock = idx_allow + 3
    head.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 0, idx_kill - 2))
    head.append(_stmt(BPF_LD | BPF_W | BPF_ABS, _OFF_NR))
    head.append(_jump(BPF_JMP | BPF_JGE | BPF_K, X32_SYSCALL_BIT, idx_kill - idx_x32 - 1, 0))
    head.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, NR["socket"], idx_sock - idx_dispatch0 - 1, 0))
    for j, nr in enumerate(nrs):
        idx = idx_dispatch0 + 1 + j
        head.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, nr, idx_eperm - idx - 1, 0))
    prog = head + [ret_allow, ret_eperm, ret_kill] + sock_block
    assert len(head) == idx_allow, (len(head), idx_allow)
    return b"".join(prog)


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def seccomp_status() -> dict:
    """``Seccomp`` mode, filter count and ``NoNewPrivs`` from /proc/self/status."""
    out = {"seccomp_mode": None, "seccomp_filters": None, "no_new_privs": None}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key, _, val = line.partition(":")
                val = val.strip()
                if key == "Seccomp":
                    out["seccomp_mode"] = int(val)
                elif key == "Seccomp_filters":
                    out["seccomp_filters"] = int(val)
                elif key == "NoNewPrivs":
                    out["no_new_privs"] = int(val)
    except OSError:
        pass
    return out


def install_filter(prog: bytes | None = None, *, tsync: bool = True) -> dict:
    """Set NO_NEW_PRIVS and install the miner filter on the calling process
    (and, with ``tsync``, every thread it already has). Children inherit it.

    Raises ``RuntimeError`` (fail-closed) on a non-x86_64 machine or when the
    kernel refuses; never returns with the process half-sandboxed.
    """
    if platform.machine() != "x86_64":
        raise RuntimeError(f"seccomp sandbox supports x86_64 only (got {platform.machine()})")
    prog = prog if prog is not None else build_filter()
    libc = _libc()
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        e = ctypes.get_errno()
        raise RuntimeError(f"PR_SET_NO_NEW_PRIVS failed: {os.strerror(e)}")
    buf = ctypes.create_string_buffer(prog, len(prog))
    fprog = _SockFprog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))
    flags = SECCOMP_FILTER_FLAG_TSYNC if tsync else 0
    rc = libc.syscall(NR_SECCOMP, SECCOMP_SET_MODE_FILTER, flags, ctypes.byref(fprog))
    if rc != 0:
        e = ctypes.get_errno()
        rc2 = libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0)
        if rc2 != 0:
            e2 = ctypes.get_errno()
            raise RuntimeError(f"seccomp install failed: seccomp(2) {os.strerror(e)}; "
                               f"prctl {os.strerror(e2)}")
    st = seccomp_status()
    if st.get("seccomp_mode") != SECCOMP_MODE_FILTER or st.get("no_new_privs") != 1:
        raise RuntimeError(f"seccomp install not reflected in /proc/self/status: {st}")
    return st


def probe_network_denied() -> dict:
    """Positive probe: an AF_INET/AF_INET6 socket must be refused with EPERM
    and an AF_UNIX socket must still work. Returns the observations; the
    caller decides (``net_denied`` is the load-bearing bit)."""
    obs = {"inet": None, "inet6": None, "unix": None}
    for name, fam in (("inet", socket.AF_INET), ("inet6", socket.AF_INET6)):
        try:
            s = socket.socket(fam, socket.SOCK_STREAM)
            s.close()
            obs[name] = "open"
        except PermissionError:
            obs[name] = "eperm"
        except OSError as e:
            obs[name] = f"oserror:{e.errno}"
    try:
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        a.close(); b.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.close()
        obs["unix"] = "ok"
    except OSError as e:
        obs["unix"] = f"oserror:{e.errno}"
    obs["net_denied"] = obs["inet"] == "eperm" and obs["inet6"] == "eperm"
    obs["unix_ok"] = obs["unix"] == "ok"
    return obs


def engage(*, require_probe: bool = True) -> dict:
    """Install the filter and prove it. Returns the attestation dict the worker
    embeds in its authenticated output. Raises on any failure."""
    st = install_filter()
    probe = probe_network_denied()
    att = {
        "sandbox_mode": SANDBOX_MODE_SECCOMP,
        "seccomp_mode": st.get("seccomp_mode"),
        "seccomp_filters": st.get("seccomp_filters"),
        "no_new_privs": st.get("no_new_privs"),
        "net_probe": probe,
        "net_denied": bool(probe.get("net_denied")),
    }
    if require_probe and not (att["net_denied"] and probe.get("unix_ok")):
        raise RuntimeError(f"seccomp sandbox probe failed: {probe}")
    return att


def attestation_ok(result: dict) -> bool:
    """Parent-side check of a worker's seccomp attestation (fields above)."""
    return (
        result.get("sandbox_mode") == SANDBOX_MODE_SECCOMP
        and result.get("worker_uid") == 65534
        and result.get("seccomp_mode") == SECCOMP_MODE_FILTER
        and result.get("no_new_privs") == 1
        and result.get("net_denied") is True
    )


def _self_test() -> int:
    """``python -m sn125.seccomp_sandbox``: install in THIS process and probe.
    Used by the orchestrator's container guard on a fresh box and by tests
    (run it in a subprocess — the filter is irreversible)."""
    import json
    import sys
    try:
        att = engage()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1
    import subprocess
    child = subprocess.run([sys.executable, "-c",
                            "import socket\n"
                            "try:\n socket.socket(); print('CHILD_NET_OPEN')\n"
                            "except PermissionError:\n print('CHILD_NET_DENIED')"],
                           capture_output=True, text=True, timeout=60)
    att["child_inherits"] = "CHILD_NET_DENIED" in child.stdout
    att["ok"] = bool(att["net_denied"] and att["child_inherits"])
    print(json.dumps(att))
    return 0 if att["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
