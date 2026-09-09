"""Auto-update supervisor for the validator: ``python -m sn125 validate --auto-update ...``.

Wraps the ordinary ``validate`` command (identical flags) in a small parent
process that keeps the checkout current with its git upstream WITHOUT ever
interrupting a round:

    supervisor                                   validator child
    ----------                                   ---------------
    fetch upstream every --update-interval s     runs rounds as usual
    new commits?  -> touch <restart flag>        ... at the round boundary, after
                                                 checkpointing state, sees the flag,
                                                 exits EXIT_RESTART (75)
    apply update  -> git reset --hard <target>   (nothing running)
                     keep operator-edited files
                     pip install -r requirements.txt   (only if it changed)
                     import smoke test; roll back on failure
    re-exec self (new supervisor code), spawn child again
                                                 resumes from validator_state.json
                                                 + the durable payment ledger

What is and is not touched by an update:

- Tracked files move to the upstream commit. Locally MODIFIED tracked files
  listed in ``SN125_UPDATE_KEEP`` (default ``sn125/config.py``) are re-applied
  on top (3-way via git stash); if that conflicts the operator's version wins
  and the upstream version is left at ``<state-dir>/backup/`` for a manual merge.
- Untracked / ignored files are never touched: the virtualenv, ``.env``,
  ``sn125/rounds/``, ``sn125/audit/`` (ledger, state checkpoint), wallets,
  ``~/.sn125``. There is no ``git clean`` anywhere in this module.
- ``pip install`` runs with THIS interpreter (``sys.executable``), i.e. inside
  the venv the supervisor was started from, and only when requirements.txt
  differs between the old and new commit.
- A failed pip install or import smoke rolls the checkout back to the previous
  commit (same protected-file handling) and the validator restarts on the old
  code; the update is retried only when upstream moves again.

Crash policy: a child that exits non-zero for any other reason is restarted
after a backoff (30 s doubling to 10 min); a pending update is applied before
the restart, so a broken release is replaced by the fix as soon as it lands.
A clean exit (0, e.g. a finite ``max_rounds`` run) stops the supervisor.

Urgent updates (``UPDATE_NOW``): a push that adds or changes the file
``UPDATE_NOW`` at the repository root, or whose commit message contains
``[update-now]``, is applied without waiting for the round boundary. The flag
is written with a ``now`` line; the validator abandons the round in progress
at its next window poll (<= 60 s) or, if it is blocked inside an evaluation,
the supervisor sends SIGINT after ``SN125_UPDATE_NOW_GRACE_S`` (default 120 s)
and SIGTERM 60 s later. The abandoned round's audit and state checkpoint are
still written; the credits debited in it are returned to the miners when the
validator starts again (roundsm.live._refund_abandoned_rounds). GPU spend on
the in-flight evaluations is the operator's cost. Rounds already published are
never affected. The trigger is edge-based (the file must CHANGE between the
running commit and the pushed one), so leaving ``UPDATE_NOW`` in the tree is
harmless; bump its contents to trigger again.

Why not Watchtower: it replaces the container the moment a new image appears,
which is mid-round almost always (rounds are ~24 h), throwing away the
in-flight evaluations and GPU spend. The safe restart point is only known to
the validator itself, hence the flag-file protocol above. The same protocol
works inside a container whose checkout is a mounted volume.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

log = logging.getLogger("sn125.autoupdate")

EXIT_RESTART = 75
RESTART_FLAG_ENV = "SN125_RESTART_FLAG"

DEFAULT_INTERVAL_S = 300.0
INTERVAL_ENV = "SN125_UPDATE_INTERVAL_S"
REMOTE_ENV = "SN125_UPDATE_REMOTE"
BRANCH_ENV = "SN125_UPDATE_BRANCH"
KEEP_ENV = "SN125_UPDATE_KEEP"
STATE_DIR_ENV = "SN125_AUTOUPDATE_DIR"
DEFAULT_KEEP = ("sn125/config.py",)
DEFAULT_STATE_DIR = Path.home() / ".sn125" / "autoupdate"
SUPERVISOR_FLAGS = ("--auto-update", "--update-interval")

_CRASH_BACKOFF_MIN_S = 30.0
_CRASH_BACKOFF_MAX_S = 600.0

UPDATE_NOW_FILE = "UPDATE_NOW"
UPDATE_NOW_TAG = "[update-now]"
RESTART_NOW_MARKER = "now"
NOW_GRACE_ENV = "SN125_UPDATE_NOW_GRACE_S"
DEFAULT_NOW_GRACE_S = 120.0
_NOW_TERM_AFTER_S = 60.0
_NOW_KILL_AFTER_S = 30.0
_SMOKE_MODULES = ("sn125.__main__", "sn125.roundsm.live", "sn125.autoupdate")


class UpdateError(Exception):
    pass


def repo_root() -> Path:
    """The checkout this package was imported from (parent of ``sn125/``)."""
    return Path(__file__).resolve().parents[1]


def strip_supervisor_flags(argv: Sequence[str]) -> list[str]:
    """Remove ``--auto-update`` / ``--update-interval X`` (both spellings) so the
    remaining argv is exactly what a plain ``validate`` would take."""
    out: list[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--auto-update":
            continue
        if a == "--update-interval":
            skip = True
            continue
        if a.startswith("--update-interval="):
            continue
        out.append(a)
    return out


def audit_dir_from_argv(argv: Sequence[str], root: str | Path | None = None) -> Path:
    """The child validator's audit directory: ``--audit-dir`` from its argv,
    else ``SN125_AUDIT_DIR``, else ``<root>/sn125/audit`` (the live loop's
    default). Used only to place the supervisor's own log file."""
    args = list(argv)
    for i, a in enumerate(args):
        if a == "--audit-dir" and i + 1 < len(args) and args[i + 1]:
            return Path(args[i + 1]).expanduser()
        if a.startswith("--audit-dir="):
            value = a.split("=", 1)[1]
            if value:
                return Path(value).expanduser()
    env_dir = (os.environ.get("SN125_AUDIT_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(root or repo_root()) / "sn125" / "audit"



class Repo:
    """Thin git wrapper over one working tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _env() -> dict[str, str]:
        env = dict(os.environ)
        for key, value in (("GIT_AUTHOR_NAME", "sn125-autoupdate"),
                           ("GIT_AUTHOR_EMAIL", "autoupdate@sn125.invalid"),
                           ("GIT_COMMITTER_NAME", "sn125-autoupdate"),
                           ("GIT_COMMITTER_EMAIL", "autoupdate@sn125.invalid")):
            env.setdefault(key, value)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        return env

    def git(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(["git", "-C", str(self.root), *args],
                              text=True, capture_output=True, env=self._env())
        if check and proc.returncode != 0:
            raise UpdateError(f"git {' '.join(args)} failed ({proc.returncode}): "
                              f"{proc.stderr.strip() or proc.stdout.strip()}")
        return proc.stdout.strip()

    def is_repo(self) -> bool:
        try:
            return self.git("rev-parse", "--is-inside-work-tree") == "true"
        except UpdateError:
            return False

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def branch(self) -> str:
        name = self.git("rev-parse", "--abbrev-ref", "HEAD")
        if name == "HEAD":
            raise UpdateError("detached HEAD: set SN125_UPDATE_BRANCH or check out a branch")
        return name

    def fetch(self, remote: str, branch: str) -> str:
        """Fetch ``remote/branch`` and return its commit sha."""
        self.git("fetch", "--quiet", "--prune", remote, branch)
        return self.git("rev-parse", "FETCH_HEAD")

    def modified_tracked(self) -> list[str]:
        """Tracked paths with uncommitted changes (worktree or index)."""
        proc = subprocess.run(["git", "-C", str(self.root), "status", "--porcelain",
                               "-z", "--untracked-files=no"],
                              capture_output=True, env=self._env())
        if proc.returncode != 0:
            raise UpdateError(f"git status failed: {proc.stderr.decode(errors='replace').strip()}")
        fields = proc.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        paths: list[str] = []
        i = 0
        while i < len(fields):
            entry = fields[i]
            i += 1
            if len(entry) < 4:
                continue
            status, path = entry[:2], entry[3:]
            if "R" in status or "C" in status:
                i += 1
            if status.strip():
                paths.append(path)
        return paths

    def changed_between(self, a: str, b: str) -> list[str]:
        out = self.git("diff", "--name-only", a, b)
        return [p for p in out.splitlines() if p]

    def log_messages(self, a: str, b: str) -> str:
        """Subject+body of every commit reachable from ``b`` but not ``a``."""
        return self.git("log", "--format=%s%n%b", f"{a}..{b}", check=False)

    def is_ancestor(self, a: str, b: str) -> bool:
        return subprocess.run(["git", "-C", str(self.root), "merge-base",
                               "--is-ancestor", a, b], capture_output=True,
                              env=self._env()).returncode == 0

    def short(self, sha: str) -> str:
        return sha[:10]


@dataclass
class UpdateResult:
    ok: bool
    old: str
    new: str
    pip_ran: bool = False
    kept: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    message: str = ""


def _backup(repo: Repo, paths: list[str], backup_root: Path, tag: str) -> Path:
    dest = backup_root / tag
    for rel in paths:
        src = repo.root / rel
        if src.is_file():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    return dest


def checkout_keeping(repo: Repo, target: str, keep: Sequence[str],
                     backup_root: Path) -> tuple[list[str], list[str]]:
    """Move the worktree to ``target`` while re-applying local edits to ``keep``.

    Returns ``(kept, conflicts)``: files whose local edits were re-applied
    cleanly, and files where the operator's version was restored verbatim
    because the upstream change conflicted (upstream copy left in the backup).
    Untracked files are never touched.
    """
    modified = set(repo.modified_tracked())
    protected = [p for p in keep if p in modified]
    unprotected = sorted(modified - set(protected))
    if unprotected:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = _backup(repo, unprotected, backup_root, f"{stamp}-{repo.short(repo.head())}-unlisted")
        log.warning("locally modified tracked files not in %s will be replaced by the "
                    "update; copies kept at %s: %s", KEEP_ENV, dest, ", ".join(unprotected))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = f"{stamp}-{repo.short(repo.head())}"
    if protected:
        _backup(repo, protected, backup_root, tag)
        repo.git("stash", "push", "--quiet", "--", *protected)
    try:
        repo.git("reset", "--hard", "--quiet", target)
    except UpdateError:
        if protected:
            repo.git("stash", "pop", "--quiet", check=False)
        raise
    kept: list[str] = []
    conflicts: list[str] = []
    if protected:
        popped = subprocess.run(["git", "-C", str(repo.root), "stash", "pop", "--quiet"],
                                text=True, capture_output=True, env=repo._env())
        if popped.returncode == 0:
            kept = list(protected)
        else:
            for rel in protected:
                try:
                    upstream_text = repo.git("show", f"{target}:{rel}")
                except UpdateError:
                    upstream_text = None
                if upstream_text is not None:
                    up_copy = backup_root / tag / "upstream" / rel
                    up_copy.parent.mkdir(parents=True, exist_ok=True)
                    up_copy.write_text(upstream_text + "\n", encoding="utf-8")
                repo.git("checkout", "stash@{0}", "--", rel)
                repo.git("reset", "--quiet", "--", rel, check=False)
            repo.git("stash", "drop", "--quiet", check=False)
            conflicts = list(protected)
            log.error("local edits to %s conflict with the update; the OPERATOR'S "
                      "version was kept verbatim and the upstream version saved under "
                      "%s for a manual merge", ", ".join(protected), backup_root / tag / "upstream")
    return kept, conflicts


def default_pip_install(root: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r",
                    str(root / "requirements.txt")], check=True, cwd=str(root))


def default_import_smoke(root: Path) -> None:
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({m!r})" for m in _SMOKE_MODULES)
    env = {**os.environ, "PYTHONPATH": str(root)}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root), env=env,
                   timeout=300)


def apply_update(repo: Repo, target: str, *, keep: Sequence[str] = DEFAULT_KEEP,
                 state_dir: Path = DEFAULT_STATE_DIR,
                 pip_install: Callable[[Path], None] = default_pip_install,
                 import_smoke: Callable[[Path], None] = default_import_smoke) -> UpdateResult:
    """Move the checkout to ``target``; install requirements if they changed;
    smoke-import; roll back to the previous commit on any failure."""
    old = repo.head()
    if old == target:
        return UpdateResult(True, old, target, message="already current")
    backup_root = state_dir / "backup"
    backup_root.mkdir(parents=True, exist_ok=True)
    reqs_changed = "requirements.txt" in repo.changed_between(old, target)
    kept, conflicts = checkout_keeping(repo, target, keep, backup_root)
    result = UpdateResult(False, old, target, kept=kept, conflicts=conflicts)
    try:
        if reqs_changed:
            log.info("requirements.txt changed in %s..%s; installing into %s",
                     repo.short(old), repo.short(target), sys.executable)
            pip_install(repo.root)
            result.pip_ran = True
        import_smoke(repo.root)
    except Exception as e:
        log.error("update %s..%s failed verification (%s); rolling back",
                  repo.short(old), repo.short(target), e)
        try:
            checkout_keeping(repo, old, keep, backup_root)
        except UpdateError as e2:
            result.message = f"update failed ({e}) AND rollback failed ({e2})"
            return result
        result.message = f"rolled back: {e}"
        return result
    result.ok = True
    result.message = "updated"
    return result



def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def keep_list() -> list[str]:
    raw = os.environ.get(KEEP_ENV)
    if raw is None:
        return list(DEFAULT_KEEP)
    return [p.strip() for p in raw.split(",") if p.strip()]


def state_dir() -> Path:
    return Path(os.environ.get(STATE_DIR_ENV, "") or DEFAULT_STATE_DIR).expanduser()


class Supervisor:
    """Runs the validator child and the update loop. See module docstring."""

    def __init__(self, child_argv: Sequence[str], *, repo: Repo | None = None,
                 interval_s: float | None = None, remote: str | None = None,
                 branch: str | None = None, keep: Sequence[str] | None = None,
                 state_dir_path: Path | None = None,
                 child_cmd: Sequence[str] | None = None,
                 self_exec: Callable[[], None] | None = None,
                 pip_install: Callable[[Path], None] = default_pip_install,
                 import_smoke: Callable[[Path], None] = default_import_smoke,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.repo = repo or Repo(repo_root())
        self.child_argv = list(child_argv)
        self.interval_s = max(5.0, float(interval_s if interval_s is not None
                                         else _env_float(INTERVAL_ENV, DEFAULT_INTERVAL_S)))
        self.remote = remote or os.environ.get(REMOTE_ENV, "") or "origin"
        self.branch = branch or os.environ.get(BRANCH_ENV, "") or None
        self.keep = list(keep) if keep is not None else keep_list()
        self.state_dir = state_dir_path or state_dir()
        self.child_cmd = list(child_cmd) if child_cmd else [sys.executable, "-m", "sn125"]
        self.self_exec = self_exec
        self.pip_install = pip_install
        self.import_smoke = import_smoke
        self.sleep = sleep
        self.flag = self.state_dir / "restart.requested"
        self.now_grace_s = max(0.0, _env_float(NOW_GRACE_ENV, DEFAULT_NOW_GRACE_S))
        self._proc: subprocess.Popen | None = None
        self._stopping = False
        self._pending: str | None = None
        self._pending_now = False
        self._now_requested_at: float | None = None
        self._now_signals_sent = 0
        self._last_fetch = 0.0
        self._fetch_failures = 0

    def _branch(self) -> str:
        return self.branch or self.repo.branch()

    def check_upstream(self) -> str | None:
        """Fetch; return the target sha when upstream is ahead of HEAD, else None."""
        try:
            target = self.repo.fetch(self.remote, self._branch())
            self._fetch_failures = 0
        except UpdateError as e:
            self._fetch_failures += 1
            if self._fetch_failures in (1, 10, 100) or self._fetch_failures % 1000 == 0:
                log.warning("update check failed (%d in a row): %s", self._fetch_failures, e)
            return None
        head = self.repo.head()
        if target == head or self.repo.is_ancestor(target, head):
            return None
        return target

    def apply(self, target: str) -> UpdateResult:
        result = apply_update(self.repo, target, keep=self.keep, state_dir=self.state_dir,
                              pip_install=self.pip_install, import_smoke=self.import_smoke)
        if result.ok:
            log.info("updated %s -> %s (%s%s%s)", self.repo.short(result.old),
                     self.repo.short(result.new),
                     "requirements installed; " if result.pip_ran else "",
                     f"kept local edits to {', '.join(result.kept)}; " if result.kept else "",
                     f"CONFLICTS kept operator version: {', '.join(result.conflicts)}"
                     if result.conflicts else "no protected-file conflicts")
        else:
            log.error("update %s -> %s NOT applied: %s", self.repo.short(result.old),
                      self.repo.short(result.new), result.message)
        return result

    def _spawn(self) -> subprocess.Popen:
        env = {**os.environ, RESTART_FLAG_ENV: str(self.flag)}
        cmd = [*self.child_cmd, *self.child_argv]
        log.info("starting validator at %s: %s", self.repo.short(self.repo.head()),
                 " ".join(cmd))
        return subprocess.Popen(cmd, cwd=str(self.repo.root), env=env)

    def _forward_signal(self, signum, _frame) -> None:
        self._stopping = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.send_signal(signum)
            except OSError:
                pass

    def _clear_flag(self) -> None:
        try:
            self.flag.unlink()
        except FileNotFoundError:
            pass

    def is_urgent(self, head: str, target: str) -> bool:
        """True when the pushed range asks to skip the round boundary: the
        ``UPDATE_NOW`` file was added/changed, or a commit message carries
        ``[update-now]``."""
        try:
            if UPDATE_NOW_FILE in self.repo.changed_between(head, target):
                return True
            return UPDATE_NOW_TAG in self.repo.log_messages(head, target).lower()
        except UpdateError as e:
            log.warning("could not inspect %s..%s for an urgent marker: %s",
                        self.repo.short(head), self.repo.short(target), e)
            return False

    def _request_restart(self, target: str, *, now: bool = False) -> None:
        if self._pending == target and (self._pending_now or not now):
            return
        self._pending = target
        self.state_dir.mkdir(parents=True, exist_ok=True)
        body = target + "\n" + (RESTART_NOW_MARKER + "\n" if now else "")
        self.flag.write_text(body, encoding="utf-8")
        if now:
            self._pending_now = True
            self._now_requested_at = time.monotonic()
            self._now_signals_sent = 0
            log.warning("URGENT update available %s -> %s (%s); the validator abandons "
                        "the round in progress and restarts now (SIGINT after %.0fs "
                        "if it is blocked in an evaluation)",
                        self.repo.short(self.repo.head()), self.repo.short(target),
                        UPDATE_NOW_FILE, self.now_grace_s)
        else:
            log.info("update available %s -> %s; the validator will restart at its "
                     "next round boundary", self.repo.short(self.repo.head()),
                     self.repo.short(target))

    def _escalate_now(self, proc: subprocess.Popen) -> None:
        """After an urgent request: SIGINT once the grace period is over, then
        SIGTERM, then SIGKILL, so an evaluation-blocked child cannot pin the
        old code indefinitely."""
        if self._now_requested_at is None or proc.poll() is not None:
            return
        waited = time.monotonic() - self._now_requested_at
        steps = ((self.now_grace_s, signal.SIGINT, "SIGINT"),
                 (self.now_grace_s + _NOW_TERM_AFTER_S, signal.SIGTERM, "SIGTERM"),
                 (self.now_grace_s + _NOW_TERM_AFTER_S + _NOW_KILL_AFTER_S,
                  signal.SIGKILL, "SIGKILL"))
        if self._now_signals_sent < len(steps):
            due, sig, name = steps[self._now_signals_sent]
            if waited >= due:
                log.warning("validator still running %.0fs after the urgent request; "
                            "sending %s", waited, name)
                try:
                    proc.send_signal(sig)
                except OSError:
                    pass
                self._now_signals_sent += 1

    def _update_before_start(self) -> bool:
        """Apply a pending/available update with nothing running. Returns True
        when the checkout changed (caller should re-exec)."""
        target = self.check_upstream()
        if target is None:
            return False
        result = self.apply(target)
        return result.ok

    def run(self, install_signal_handlers: bool = True) -> int:
        if not self.repo.is_repo():
            log.error("%s is not a git checkout; --auto-update needs a clone of the "
                      "release repository (git clone <repo>; pip install -r requirements.txt "
                      "in your venv; run from that directory)", self.repo.root)
            return 2
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._clear_flag()
        if install_signal_handlers:
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._forward_signal)
        log.info("auto-update supervisor: %s branch %s, every %.0fs, keeping %s, "
                 "state in %s", self.remote, self._branch(), self.interval_s,
                 ", ".join(self.keep) or "(nothing)", self.state_dir)
        backoff = _CRASH_BACKOFF_MIN_S
        while not self._stopping:
            if self._update_before_start():
                self._pending = None
                if self.self_exec is not None:
                    log.info("re-executing the supervisor on the updated code")
                    self.self_exec()
                    return 0
            self._pending = None
            self._pending_now = False
            self._now_requested_at = None
            self._clear_flag()
            self._proc = self._spawn()
            rc = self._monitor(self._proc)
            self._proc = None
            if rc == EXIT_RESTART:
                log.info("validator stopped %s for restart",
                         "mid-round (urgent update)" if self._pending_now
                         else "at a round boundary")
                backoff = _CRASH_BACKOFF_MIN_S
                continue
            if self._pending_now and not self._stopping:
                log.warning("validator exited with %s after the urgent request; "
                            "applying the update and restarting now", rc)
                backoff = _CRASH_BACKOFF_MIN_S
                continue
            if self._stopping:
                return 0 if rc in (0, -signal.SIGINT, -signal.SIGTERM, 130, 143) else rc
            if rc == 0:
                log.info("validator exited cleanly; supervisor done")
                return 0
            log.error("validator exited with %s; restarting in %.0fs "
                      "(a pending update is applied first)", rc, backoff)
            self._sleep_watching(backoff)
            backoff = min(backoff * 2, _CRASH_BACKOFF_MAX_S)
        return 0

    def _monitor(self, proc: subprocess.Popen) -> int:
        next_check = time.monotonic() + self.interval_s
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + self.interval_s
                target = self.check_upstream()
                if target is not None:
                    urgent = self._pending_now or self.is_urgent(self.repo.head(), target)
                    self._request_restart(target, now=urgent)
            if self._pending_now:
                self._escalate_now(proc)
            self.sleep(min(1.0, self.interval_s))

    def _sleep_watching(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stopping and time.monotonic() < end:
            self.sleep(min(1.0, end - time.monotonic()))


def supervise(full_argv: Sequence[str], *, interval_s: float | None = None) -> int:
    """Entry point used by ``python -m sn125 validate --auto-update ...``.

    ``full_argv`` is ``sys.argv[1:]`` (includes the supervisor flags); the child
    gets the same argv minus those flags. After a successful update the
    supervisor re-executes itself with the ORIGINAL argv so the new supervisor
    code runs too.
    """
    child_argv = strip_supervisor_flags(full_argv)
    root = repo_root()
    program = [sys.executable, "-m", "sn125", *full_argv]
    try:
        from .logfile import attach_rotating_log
        attach_rotating_log(audit_dir_from_argv(child_argv, root) / "autoupdate.log")
    except Exception as exc:
        log.warning("supervisor file log unavailable: %s", exc)

    def self_exec() -> None:
        os.chdir(str(root))
        os.execv(sys.executable, program)

    sup = Supervisor(child_argv, repo=Repo(root), interval_s=interval_s, self_exec=self_exec)
    return sup.run()
