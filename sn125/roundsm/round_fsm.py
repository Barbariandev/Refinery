"""Round state machine (DESIGN.md §3.6, §8, E5 batched rounds, §2.4 pause gates).

Phases::

    SETUP -> COMMIT_OPEN -> REVEAL_OPEN -> EVALUATING -> PUBLISHED
                 |               |             |
                 +---- PAUSED (outage gate, resumes to the stored phase) ----+

Structural properties (each is a threat-model row, and each has a test):

- **E5 (no sniping):** all commits close before any reveal is accepted, and
  all reveals close before any evaluation starts — enforced by phase order,
  not by scheduling discipline. A copied commit hash is useless without the
  preimage (reveal must SHA-256 to the commit).
- **E4 (payment gate):** a commit is accepted iff the hotkey's coldkey holds
  >= 1 prepaid credit; the credit is debited at acceptance. An unrevealed
  commit is consumed — no refund.
- **§2.4 (infra adversity promoted to protocol):** provisioning flakes are $0
  and relaunch the same commitment; 3 *consecutive* flakes = outage and the
  round PAUSES — it never silently shrinks (deadlines stretch by exactly the
  paused time, enforced because close_*() refuses to fire before the
  pause-extended deadline). A run crash relaunches once; a second crash on
  the same run = infra-DQ and the credit is refunded.
- **Fee pin ordering (§8):** open_commits() refuses to run unless the
  registry has this round's fee pinned, and pinned to the configured value —
  the pin is published before the window opens by construction.

Time is injected (``clock``) so every timing rule is deterministic in tests.
The validator loop drives transitions; the FSM enforces legality.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from sn125.payments.registry import PaymentRegistry


class RoundError(Exception):
    pass


class PhaseError(RoundError):
    """Operation illegal in the current phase."""


class Phase(Enum):
    SETUP = "setup"
    COMMIT_OPEN = "commit_open"
    REVEAL_OPEN = "reveal_open"
    EVALUATING = "evaluating"
    PUBLISHED = "published"
    PAUSED = "paused"


class SubStatus(Enum):
    COMMITTED = "committed"
    UNREVEALED = "unrevealed"
    REVEALED = "revealed"
    RUNNING = "running"
    SCORED = "scored"
    DQ = "dq"
    INFRA_DQ = "infra_dq"
    DEFERRED = "deferred"

    @property
    def terminal(self) -> bool:
        return self in (SubStatus.UNREVEALED, SubStatus.SCORED,
                        SubStatus.DQ, SubStatus.INFRA_DQ, SubStatus.DEFERRED)


@dataclass(frozen=True)
class RoundConfig:
    """Pinned, published per round (dashboard §9.1 'round config')."""
    round_id: str
    scale: str
    horizon_steps: int
    task_family: str
    gpu_sku: str
    fee_rao: int
    commit_window_s: float
    reveal_window_s: float


@dataclass
class Submission:
    commit_hash: str
    hotkey: str
    coldkey: str
    status: SubStatus = SubStatus.COMMITTED
    payload: bytes | None = None
    accepted_at: float = 0.0
    accept_index: int = 0
    crashes: int = 0
    relaunched: bool = False
    score: float | None = None
    deferrals: int = 0
    selected_for_eval: bool = False
    selection_rank: int | None = None
    forensics: list[str] = field(default_factory=list)


def commit_hash_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


OUTAGE_FLAKE_STREAK = 3

MAX_EVAL_PER_ROUND = 8


class RoundFSM:
    def __init__(self, config: RoundConfig, registry: PaymentRegistry,
                 clock: Callable[[], float]) -> None:
        self.config = config
        self.registry = registry
        self._clock = clock
        self.phase = Phase.SETUP
        self.submissions: dict[str, Submission] = {}
        self._phase_started_at: float | None = None
        self._paused_in_phase = 0.0
        self._paused_at: float | None = None
        self._resume_phase: Phase | None = None
        self.pause_reasons: list[str] = []
        self._flake_streak = 0
        self._selected = False
        self._accept_seq = 0

    def _require(self, *phases: Phase) -> None:
        if self.phase not in phases:
            raise PhaseError(
                f"illegal in phase {self.phase.value} "
                f"(needs {'/'.join(p.value for p in phases)})")

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_started_at = self._clock()
        self._paused_in_phase = 0.0

    def deadline(self) -> float | None:
        """Pause-extended deadline of the current windowed phase (§2.4:
        pausing stretches the window — a round never silently shrinks)."""
        if self._phase_started_at is None:
            return None
        if self.phase is Phase.COMMIT_OPEN:
            window = self.config.commit_window_s
        elif self.phase is Phase.REVEAL_OPEN:
            window = self.config.reveal_window_s
        else:
            return None
        return self._phase_started_at + window + self._paused_in_phase

    def open_commits(self) -> None:
        self._require(Phase.SETUP)
        pinned = self.registry.fee_for_round(self.config.round_id)
        if pinned is None:
            raise RoundError("fee not pinned: pin must publish before the "
                             "commit window opens (§8)")
        if pinned != self.config.fee_rao:
            raise RoundError(f"fee pin {pinned} != round config {self.config.fee_rao}")
        self._enter(Phase.COMMIT_OPEN)

    def accept_commit(self, hotkey: str, commit_hash: str) -> Submission:
        """Credit-gated (E4): debits 1 credit from the hotkey's coldkey NOW."""
        self._require(Phase.COMMIT_OPEN)
        if commit_hash in self.submissions:
            raise RoundError("duplicate commit hash")
        coldkey = self.registry.debit_for_commit(hotkey, self.config.round_id)
        self._accept_seq += 1
        sub = Submission(
            commit_hash=commit_hash,
            hotkey=hotkey,
            coldkey=coldkey,
            accepted_at=self._clock(),
            accept_index=self._accept_seq,
        )
        self.submissions[commit_hash] = sub
        return sub

    def close_commits(self) -> None:
        self._require(Phase.COMMIT_OPEN)
        if self._clock() < self.deadline():
            raise RoundError("commit window not over (pause-extended deadline)")
        self._enter(Phase.REVEAL_OPEN)

    def reveal(self, commit_hash: str, payload: bytes) -> Submission:
        self._require(Phase.REVEAL_OPEN)
        sub = self.submissions.get(commit_hash)
        if sub is None:
            raise RoundError("no such commit")
        if sub.status is not SubStatus.COMMITTED:
            raise RoundError(f"commit already {sub.status.value}")
        if commit_hash_of(payload) != commit_hash:
            raise RoundError("preimage does not hash to commit")
        sub.payload = payload
        sub.status = SubStatus.REVEALED
        return sub

    def close_reveals(self) -> None:
        self._require(Phase.REVEAL_OPEN)
        if self._clock() < self.deadline():
            raise RoundError("reveal window not over (pause-extended deadline)")
        for sub in self.submissions.values():
            if sub.status is SubStatus.COMMITTED:
                sub.status = SubStatus.UNREVEALED
                sub.forensics.append("never revealed; credit consumed")
        self._enter(Phase.EVALUATING)

    def select_for_evaluation(self, *, recent_hotkeys=(),
                              max_eval: int = MAX_EVAL_PER_ROUND
                              ) -> tuple[list[str], list[str]]:
        """§6.2 selection rule. From the REVEALED (valid, paid, gate-passing)
        commits choose up to ``max_eval`` to evaluate this round; the rest are
        marked DEFERRED — **credit retained**, rolled over FIFO to the next
        round (the loop re-queues them). Returns (selected, deferred) lists of
        commit hashes.

        Priority (ascending sort key, first wins):
          1. ``-deferrals`` — a commit rolled over more times outranks fresher
             ones (FIFO age). This is the bounded-wait guarantee: the set of
             commits with >=D deferrals strictly shrinks each round, so a paid
             commit is evaluated within a bounded number of rounds; no fee is
             ever wasted and no refund is needed.
          2. ``accepted_at`` — earliest validator-accepted commit first.
          3. ``accept_index`` — stable FIFO order when the injected clock has
             coarse resolution or tests accept several commits at the same time.
          4. ``commit_hash`` — deterministic final tiebreak for audit replay.

        ``recent_hotkeys`` is accepted for driver API compatibility, but slot
        allocation is intentionally FIFO now: first valid, gate-passing commits
        get the <=8 B200 evaluation slots.

        Idempotent within a round (raises if called twice)."""
        self._require(Phase.EVALUATING)
        if self._selected:
            raise RoundError("selection already run for this round")
        self._selected = True
        revealed = [s for s in self.submissions.values()
                    if s.status is SubStatus.REVEALED]
        revealed.sort(key=lambda s: (-s.deferrals,
                                     s.accepted_at,
                                     s.accept_index,
                                     s.commit_hash))
        selected = revealed[:max_eval]
        deferred = revealed[max_eval:]
        for rank, sub in enumerate(revealed, 1):
            sub.selection_rank = rank
            sub.selected_for_eval = sub in selected
        for sub in deferred:
            sub.status = SubStatus.DEFERRED
            sub.deferrals += 1
            sub.forensics.append(
                f"deferred (overflow >{max_eval}); credit retained, "
                f"FIFO rollover #{sub.deferrals}")
        return ([s.commit_hash for s in selected],
                [s.commit_hash for s in deferred])

    def carryover(self) -> list[Submission]:
        """The DEFERRED submissions to re-queue into the next round's FSM
        (credit already paid; re-inserted as REVEALED with deferrals preserved)."""
        return [s for s in self.submissions.values()
                if s.status is SubStatus.DEFERRED]

    def readmit_deferred(self, sub: Submission) -> Submission:
        """Re-insert a submission deferred in a prior round (§6.2 FIFO rollover).
        Credit was already debited and the preimage already revealed, so it skips
        commit/reveal and re-enters directly as REVEALED with its ``deferrals``
        count preserved (which is what gives it FIFO priority next selection).
        Called during this round's COMMIT_OPEN, before fresh commits stream in."""
        self._require(Phase.COMMIT_OPEN)
        if sub.commit_hash in self.submissions:
            raise RoundError("duplicate commit hash")
        if sub.payload is None:
            raise RoundError("carryover submission has no revealed payload")
        sub.status = SubStatus.REVEALED
        sub.forensics.append(f"re-admitted from carryover (deferrals={sub.deferrals})")
        self.submissions[sub.commit_hash] = sub
        return sub

    def start_run(self, commit_hash: str) -> None:
        self._require(Phase.EVALUATING)
        sub = self._eval_sub(commit_hash, SubStatus.REVEALED)
        sub.status = SubStatus.RUNNING

    def record_provision_ok(self) -> None:
        """A box reached RUNNING: the consecutive-flake streak resets."""
        self._flake_streak = 0

    def record_provisioning_flake(self, commit_hash: str) -> None:
        """Box never came up: $0, same commitment relaunches; 3 consecutive
        = outage -> the round pauses (never silently shrinks)."""
        self._require(Phase.EVALUATING)
        sub = self._eval_sub(commit_hash, SubStatus.REVEALED, SubStatus.RUNNING)
        sub.forensics.append("provisioning flake ($0, relaunch same commitment)")
        self._flake_streak += 1
        if self._flake_streak >= OUTAGE_FLAKE_STREAK:
            self.pause(f"{self._flake_streak} consecutive provisioning flakes "
                       "= provider outage (§2.4)")

    def record_crash(self, commit_hash: str) -> None:
        """Intermittent native crash: relaunch once; second crash on the same
        run = infra-DQ -> credit refunded (§2.4)."""
        self._require(Phase.EVALUATING)
        sub = self._eval_sub(commit_hash, SubStatus.RUNNING)
        sub.crashes += 1
        if sub.crashes == 1:
            sub.relaunched = True
            sub.forensics.append("crash #1: relaunched once")
        else:
            sub.status = SubStatus.INFRA_DQ
            sub.forensics.append("crash #2 same run: infra-DQ, credit refunded")
            self.registry.refund_credit(
                sub.coldkey, self.config.round_id,
                note=f"infra-DQ refund for commit {sub.commit_hash[:12]}")

    def record_score(self, commit_hash: str, score: float) -> None:
        self._require(Phase.EVALUATING)
        sub = self._eval_sub(commit_hash, SubStatus.RUNNING)
        sub.score = score
        sub.status = SubStatus.SCORED

    def disqualify(self, commit_hash: str, reason: str) -> None:
        """Miner-fault DQ (cap breach, replay divergence, gate violation):
        fee burned, no refund."""
        self._require(Phase.EVALUATING)
        sub = self._eval_sub(commit_hash, SubStatus.REVEALED, SubStatus.RUNNING)
        sub.status = SubStatus.DQ
        sub.forensics.append(f"DQ: {reason}")

    def infra_dq(self, commit_hash: str, reason: str) -> None:
        """Our-fault DQ after the cloud layer exhausted its own relaunch retries
        (§2.4): credit refunded. This is the high-level equivalent of the second
        same-run crash — used by the round driver when ``evaluate`` reports that a
        submission could not be scored for infrastructure reasons (not miner
        fault). ``record_crash`` remains the fine-grained per-attempt path."""
        self._require(Phase.EVALUATING)
        sub = self._eval_sub(commit_hash, SubStatus.REVEALED, SubStatus.RUNNING)
        sub.status = SubStatus.INFRA_DQ
        sub.forensics.append(f"infra-DQ (credit refunded): {reason}")
        self.registry.refund_credit(
            sub.coldkey, self.config.round_id,
            note=f"infra-DQ refund for commit {sub.commit_hash[:12]}")

    def _eval_sub(self, commit_hash: str, *statuses: SubStatus) -> Submission:
        sub = self.submissions.get(commit_hash)
        if sub is None:
            raise RoundError("no such commit")
        if sub.status not in statuses:
            raise RoundError(f"submission is {sub.status.value}, needs "
                             f"{'/'.join(s.value for s in statuses)}")
        return sub

    def pause(self, reason: str) -> None:
        if self.phase in (Phase.PAUSED, Phase.PUBLISHED, Phase.SETUP):
            raise PhaseError(f"cannot pause from {self.phase.value}")
        self._resume_phase = self.phase
        self._paused_at = self._clock()
        self.phase = Phase.PAUSED
        self.pause_reasons.append(reason)

    def resume(self) -> None:
        self._require(Phase.PAUSED)
        self._paused_in_phase += self._clock() - self._paused_at
        self.phase = self._resume_phase
        self._paused_at = None
        self._resume_phase = None
        self._flake_streak = 0

    def publish(self) -> dict:
        """All revealed work terminal -> PUBLISHED; returns the round report
        (dashboard §9.1: config, per-submission lifecycle, forensics)."""
        self._require(Phase.EVALUATING)
        open_subs = [s for s in self.submissions.values() if not s.status.terminal]
        if open_subs:
            raise RoundError(
                f"{len(open_subs)} submission(s) not terminal: "
                + ", ".join(s.commit_hash[:12] for s in open_subs))
        self._enter(Phase.PUBLISHED)
        return {
            "round_id": self.config.round_id,
            "config": {
                "scale": self.config.scale,
                "horizon_steps": self.config.horizon_steps,
                "task_family": self.config.task_family,
                "gpu_sku": self.config.gpu_sku,
                "fee_rao": self.config.fee_rao,
            },
            "pauses": list(self.pause_reasons),
            "submissions": [
                {
                    "commit_hash": s.commit_hash,
                    "hotkey": s.hotkey,
                    "status": s.status.value,
                    "score": s.score,
                    "relaunched": s.relaunched,
                    "deferrals": s.deferrals,
                    "accepted_at": s.accepted_at,
                    "accept_index": s.accept_index,
                    "selected_for_eval": s.selected_for_eval,
                    "selection_rank": s.selection_rank,
                    "forensics": list(s.forensics),
                }
                for s in self.submissions.values()
            ],
        }
