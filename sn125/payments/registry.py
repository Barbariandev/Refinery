"""PaymentRegistry (DESIGN.md §8) — prepaid evaluation credits, on-chain auditable.

Rules implemented (each maps to a §8 / E4 clause):

- Credits are granted on **observed finalized transfers** to the treasury
  coldkey: ``floor(amount / round_fee)`` credits to the *sender coldkey*.
  Per-transfer floor — two half-fee payments do NOT sum to a credit (the
  remainder is recorded in the ledger for auditability, never credited).
- Credits are spendable by **any hotkey under that coldkey** (Sybil-neutral:
  still one fee per submission). Hotkey→coldkey resolution comes from the
  chain's metagraph association, never from miner claims.
- Debit happens at **commit acceptance** (an unrevealed commit still consumed
  the slot — E4 commit-spam).
- Infra-DQ refunds a **credit**, never TAO (§2.4 / E4: no refund-fraud surface).
- The TAO fee is **pinned per round** before the commit window opens and is
  immutable within it (no intra-round oracle gaming). Grants use the pin in
  force when the transfer is processed.
- The ledger is an append-only event list; ``replay()`` reconstructs every
  balance from events alone, which is the property that makes the published
  ledger independently auditable (E6).

Chain I/O is behind ``ChainView`` — a read-only protocol (§5.3: the payment
watcher holds no keys). ``MockChain`` implements it with explicit finality so
tests can exercise the finalized-only rule. The real implementation wraps
bittensor-lib and is injected later; nothing here imports chain libraries.

Amounts are integer **rao** (1 TAO = 1e9 rao) — no float money.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Protocol

RAO_PER_TAO = 10**9


class PaymentError(Exception):
    pass


class UnknownHotkeyError(PaymentError):
    """Hotkey has no coldkey association on the subnet metagraph."""


class InsufficientCreditError(PaymentError):
    pass


class FeeNotPinnedError(PaymentError):
    pass


@dataclass(frozen=True)
class Transfer:
    """A finalized transfer to the treasury coldkey, as seen on chain."""
    tx_id: str
    src_coldkey: str
    amount_rao: int
    block: int


class ChainView(Protocol):
    """Read-only chain access the registry needs. No keys, no writes (§5.3)."""

    def finalized_transfers_to_treasury(self, since_block: int) -> list[Transfer]:
        """Transfers with finalized block > since_block, ascending by block."""
        ...

    def hotkey_owner(self, hotkey: str) -> str | None:
        """Coldkey owning ``hotkey`` on the subnet metagraph, or None."""
        ...


class MockChain:
    """In-memory ChainView with explicit finality for tests.

    A transfer enters at the current head block and becomes visible to
    ``finalized_transfers_to_treasury`` only once ``head - block >=
    finality_depth`` — so tests can prove the registry never credits
    unfinalized money.
    """

    def __init__(self, finality_depth: int = 2) -> None:
        self.finality_depth = finality_depth
        self.head = 0
        self._transfers: list[Transfer] = []
        self._hotkey_owner: dict[str, str] = {}
        self._tx_seq = itertools.count()

    def register_hotkey(self, hotkey: str, coldkey: str) -> None:
        self._hotkey_owner[hotkey] = coldkey

    def send_to_treasury(self, src_coldkey: str, amount_rao: int,
                         tx_id: str | None = None) -> str:
        if amount_rao <= 0:
            raise ValueError("transfer amount must be positive")
        tx = Transfer(tx_id or f"tx-{next(self._tx_seq)}", src_coldkey,
                      amount_rao, self.head)
        self._transfers.append(tx)
        return tx.tx_id

    def advance_blocks(self, n: int = 1) -> None:
        self.head += n

    def finalized_transfers_to_treasury(self, since_block: int) -> list[Transfer]:
        cutoff = self.head - self.finality_depth
        return sorted(
            (t for t in self._transfers if since_block < t.block <= cutoff),
            key=lambda t: (t.block, t.tx_id),
        )

    def hotkey_owner(self, hotkey: str) -> str | None:
        return self._hotkey_owner.get(hotkey)


@dataclass(frozen=True)
class LedgerEvent:
    """One append-only ledger entry. The published ledger is exactly these."""
    seq: int
    kind: str
    round_id: str | None
    coldkey: str | None
    credits: int
    rao: int
    tx_id: str | None
    hotkey: str | None
    note: str


class PaymentRegistry:
    """Credit ledger over a read-only ChainView. All mutation is evented."""

    def __init__(self, chain: ChainView) -> None:
        self._chain = chain
        self._events: list[LedgerEvent] = []
        self._balances: dict[str, int] = {}
        self._seen_tx: set[str] = set()
        self._last_block = -1
        self._fee_rao: int | None = None
        self._pinned_rounds: dict[str, int] = {}

    def pin_fee(self, round_id: str, fee_rao: int) -> None:
        """Pin the TAO fee for a round. One pin per round, ever (§8)."""
        if fee_rao <= 0:
            raise ValueError("fee must be positive")
        if round_id in self._pinned_rounds:
            raise PaymentError(f"fee already pinned for round {round_id!r}")
        self._pinned_rounds[round_id] = fee_rao
        self._fee_rao = fee_rao
        self._append("fee_pin", round_id=round_id, rao=fee_rao,
                     note=f"fee pinned at {fee_rao} rao")

    def fee_for_round(self, round_id: str) -> int | None:
        return self._pinned_rounds.get(round_id)

    def sync(self) -> int:
        """Process newly finalized treasury transfers; returns count processed."""
        if self._fee_rao is None:
            raise FeeNotPinnedError("cannot grant credits before any fee pin")
        transfers = self._chain.finalized_transfers_to_treasury(self._last_block)
        n = 0
        for t in transfers:
            self._last_block = max(self._last_block, t.block)
            if t.tx_id in self._seen_tx:
                continue
            self._seen_tx.add(t.tx_id)
            credits = t.amount_rao // self._fee_rao
            remainder = t.amount_rao - credits * self._fee_rao
            if credits:
                self._balances[t.src_coldkey] = (
                    self._balances.get(t.src_coldkey, 0) + credits)
            self._append(
                "grant", coldkey=t.src_coldkey, credits=credits,
                rao=t.amount_rao, tx_id=t.tx_id,
                note=(f"{credits} credit(s), remainder {remainder} rao"
                      if credits else
                      f"below fee ({t.amount_rao} < {self._fee_rao} rao): 0 credits"),
            )
            n += 1
        return n

    def balance(self, coldkey: str) -> int:
        return self._balances.get(coldkey, 0)

    def coldkey_for_hotkey(self, hotkey: str) -> str:
        coldkey = self._chain.hotkey_owner(hotkey)
        if coldkey is None:
            raise UnknownHotkeyError(f"hotkey {hotkey!r} not on metagraph")
        return coldkey

    def debit_for_commit(self, hotkey: str, round_id: str) -> str:
        """Debit 1 credit from the hotkey's coldkey; returns the coldkey."""
        coldkey = self.coldkey_for_hotkey(hotkey)
        if self._balances.get(coldkey, 0) < 1:
            raise InsufficientCreditError(
                f"coldkey {coldkey!r} (hotkey {hotkey!r}) has no credit")
        self._balances[coldkey] -= 1
        self._append("debit", round_id=round_id, coldkey=coldkey, credits=-1,
                     hotkey=hotkey, note="debited at commit acceptance")
        return coldkey

    def refund_credit(self, coldkey: str, round_id: str, note: str) -> None:
        """Infra-DQ refund: a credit, never TAO (§2.4)."""
        self._balances[coldkey] = self._balances.get(coldkey, 0) + 1
        self._append("refund", round_id=round_id, coldkey=coldkey, credits=1,
                     note=note)

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    def events_json(self) -> str:
        """Publishable ledger (dashboard §9.1 'credit ledger')."""
        return json.dumps([asdict(e) for e in self._events], indent=1)

    @staticmethod
    def replay(events: list[LedgerEvent]) -> dict[str, int]:
        """Reconstruct all balances from the event log alone (E6 audit)."""
        balances: dict[str, int] = {}
        for e in events:
            if e.coldkey is not None and e.credits:
                balances[e.coldkey] = balances.get(e.coldkey, 0) + e.credits
        return {k: v for k, v in balances.items() if v or k in balances}

    def _append(self, kind: str, *, round_id: str | None = None,
                coldkey: str | None = None, credits: int = 0, rao: int = 0,
                tx_id: str | None = None, hotkey: str | None = None,
                note: str = "") -> None:
        self._events.append(LedgerEvent(
            seq=len(self._events), kind=kind, round_id=round_id,
            coldkey=coldkey, credits=credits, rao=rao, tx_id=tx_id,
            hotkey=hotkey, note=note))
