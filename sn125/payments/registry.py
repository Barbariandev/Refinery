"""PaymentRegistry (DESIGN.md §8) — prepaid evaluation credits, on-chain auditable.

Rules implemented (each maps to a §8 / E4 clause):

- Credits are granted on **observed finalized transfers** to the treasury
  coldkey: ``floor((carry + amount) / fee)`` credits to the *sender coldkey*,
  where ``carry`` is that coldkey's unspent rao remainder from earlier
  transfers. The remainder below one fee is **carried forward** (per
  coldkey, in rao) and recorded on the grant event, so an under- or
  over-payment is never silently kept by the treasury: two half-fee
  transfers buy one credit, 1.00 TAO against a 0.73 fee buys one credit and
  keeps 0.27 TAO toward the next. Carry is rao, so it is applied at whatever
  fee is in force when the next transfer lands.
- Credits are spendable by **any hotkey under that coldkey** (Sybil-neutral:
  still one fee per submission). Hotkey→coldkey resolution comes from the
  chain's metagraph association, never from miner claims.
- Debit happens at **commit acceptance** (an unrevealed commit still consumed
  the slot — E4 commit-spam).
- Infra-DQ refunds a **credit**, never TAO (§2.4 / E4: no refund-fraud surface).
- The TAO fee is **pinned per round** before the commit window opens and is
  immutable within it (no intra-round oracle gaming). Grants use the pin in
  force when the transfer is processed. ``set_grant_fee`` records the fee in
  force *between* rounds (validator start-up / backfill) so the continuous
  scanner can credit deposits that land before the first round pin.
- The ledger is an append-only event list; ``replay()`` reconstructs every
  balance from events alone, which is the property that makes the published
  ledger independently auditable (E6).

Durability: with ``store_path`` set, every mutation (pin, grant batch, debit,
refund) is written as a full atomic snapshot (store.py) BEFORE it is
considered applied; a failed write rolls the in-memory state back so memory
never runs ahead of disk. A restart reloads the events, re-derives balances by
replay (and refuses a snapshot whose stored balances disagree), restores the
processed-transfer set and the chain scan cursor, so no deposit is ever
credited twice and no spent credit comes back.

Concurrency: the continuous payment watcher (watch.py) calls ``sync`` from a
background thread while the round loop debits/refunds on the main thread.
All state is guarded by one re-entrant lock; chain I/O for ``sync`` happens
outside it (serialized by a second lock) so a slow RPC never blocks a commit.

Chain I/O is behind ``ChainView`` — a read-only protocol (§5.3: the payment
watcher holds no keys). ``MockChain`` implements it with explicit finality so
tests can exercise the finalized-only rule. Nothing here imports chain libraries.

Amounts are integer **rao** (1 TAO = 1e9 rao) — no float money.
"""
from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .store import LEDGER_VERSION, LedgerStoreError, atomic_write_json, load_ledger

log = logging.getLogger("sn125.payments")

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
    carry_rao: int = 0


class PaymentRegistry:
    """Credit ledger over a read-only ChainView. All mutation is evented.

    ``store_path`` enables durable snapshots (see module docstring). Without
    it the registry is purely in-memory (tests / MockChain dry-runs).
    """

    def __init__(self, chain: ChainView, store_path: str | Path | None = None) -> None:
        self._chain = chain
        self._lock = threading.RLock()
        self._sync_lock = threading.Lock()
        self._store_path = Path(store_path) if store_path else None
        self._events: list[LedgerEvent] = []
        self._balances: dict[str, int] = {}
        self._carry: dict[str, int] = {}
        self._seen_tx: set[str] = set()
        self._last_block = -1
        self._fee_rao: int | None = None
        self._pinned_rounds: dict[str, int] = {}
        if self._store_path is not None:
            self._load()

    @property
    def store_path(self) -> Path | None:
        return self._store_path

    @property
    def current_fee_rao(self) -> int | None:
        """Fee in force for grants right now (last pin), or None before any pin."""
        with self._lock:
            return self._fee_rao

    @property
    def last_block(self) -> int:
        """Block of the newest processed treasury transfer (-1 = none yet)."""
        with self._lock:
            return self._last_block

    def pin_fee(self, round_id: str, fee_rao: int) -> None:
        """Pin the TAO fee for a round. One pin per round, ever (§8)."""
        if fee_rao <= 0:
            raise ValueError("fee must be positive")
        with self._lock:
            if round_id in self._pinned_rounds:
                raise PaymentError(f"fee already pinned for round {round_id!r}")

            def apply() -> None:
                self._pinned_rounds[round_id] = fee_rao
                self._fee_rao = fee_rao
                self._append("fee_pin", round_id=round_id, rao=fee_rao,
                             note=f"fee pinned at {fee_rao} rao")

            self._transaction(apply)

    def set_grant_fee(self, fee_rao: int, note: str = "fee in force at validator start") -> bool:
        """Record the fee in force OUTSIDE a round (start-up / backfill).

        Lets the continuous scanner credit deposits before the first round
        pin, and re-bases grants after a restart with a new configured fee.
        No-op (returns False) when ``fee_rao`` is already the fee in force.
        Not tied to a round id, so it never collides with ``pin_fee``.
        """
        if fee_rao <= 0:
            raise ValueError("fee must be positive")
        with self._lock:
            if self._fee_rao == fee_rao:
                return False

            def apply() -> None:
                self._fee_rao = fee_rao
                self._append("fee_pin", rao=fee_rao, note=note)

            self._transaction(apply)
            return True

    def fee_for_round(self, round_id: str) -> int | None:
        with self._lock:
            return self._pinned_rounds.get(round_id)

    def sync(self) -> int:
        """Process newly finalized treasury transfers; returns count processed.

        Safe to call from a background thread concurrently with debits. The
        chain read runs outside the state lock; grants are applied (and
        persisted) atomically afterwards. On a failed apply the chain view's
        scan cursor is rolled back so the same blocks are re-read next call.
        """
        with self._sync_lock:
            with self._lock:
                if self._fee_rao is None:
                    raise FeeNotPinnedError("cannot grant credits before any fee pin")
                since = self._last_block
            cursor_before = self._chain_cursor()
            transfers = self._chain.finalized_transfers_to_treasury(since)
            if not transfers:
                self._persist_cursor_only(cursor_before)
                return 0
            with self._lock:
                try:
                    return self._transaction(lambda: self._apply_grants(transfers))
                except Exception:
                    self._restore_chain_cursor(cursor_before)
                    raise

    def sync_to_head(self, max_calls: int = 100_000) -> int:
        """Call ``sync`` until the chain view reports ``caught_up()``.

        Chain views without ``caught_up`` (MockChain) get exactly one call.
        Stops early when a call makes no scan progress (persistent RPC
        failure) so the caller's cadence, not a tight loop, drives retries.
        Returns the total number of transfers processed.
        """
        total = 0
        caught_up = getattr(self._chain, "caught_up", None)
        for _ in range(max(1, int(max_calls))):
            before = self._chain_cursor()
            total += self.sync()
            if caught_up is None or caught_up():
                break
            if before is not None and self._chain_cursor() == before:
                break
        return total

    def rewind_scan(self, block: int) -> None:
        """Re-scan the chain from ``block`` (inclusive) on the next ``sync``.

        Safe: already-processed transfers are skipped via their tx ids, so a
        rewind can only ADD credits for deposits that were missed.
        """
        block = max(0, int(block))
        with self._lock:
            def apply() -> None:
                self._last_block = min(self._last_block, block - 1)
                self._restore_chain_cursor(block - 1)

            self._transaction(apply)

    def _apply_grants(self, transfers: list[Transfer]) -> int:
        fee = self._fee_rao
        assert fee is not None
        n = 0
        for t in transfers:
            self._last_block = max(self._last_block, t.block)
            if t.tx_id in self._seen_tx:
                continue
            self._seen_tx.add(t.tx_id)
            carried_in = self._carry.get(t.src_coldkey, 0)
            pool = carried_in + t.amount_rao
            credits = pool // fee
            carry = pool - credits * fee
            if credits:
                self._balances[t.src_coldkey] = (
                    self._balances.get(t.src_coldkey, 0) + credits)
            if carry:
                self._carry[t.src_coldkey] = carry
            else:
                self._carry.pop(t.src_coldkey, None)
            src = (f"{t.amount_rao} rao + {carried_in} rao carried in"
                   if carried_in else f"{t.amount_rao} rao")
            self._append(
                "grant", coldkey=t.src_coldkey, credits=credits,
                rao=t.amount_rao, tx_id=t.tx_id, carry_rao=carry,
                note=(f"{src} at fee {fee} rao: {credits} credit(s), "
                      f"{carry} rao carried forward"
                      if credits else
                      f"{src} below fee {fee} rao: 0 credits, "
                      f"{carry} rao carried forward"),
            )
            n += 1
        return n

    def balance(self, coldkey: str) -> int:
        with self._lock:
            return self._balances.get(coldkey, 0)

    def balances(self) -> dict[str, int]:
        """Snapshot of every non-zero balance."""
        with self._lock:
            return {k: v for k, v in self._balances.items() if v}

    def carry_rao(self, coldkey: str) -> int:
        """Unspent rao this coldkey has paid toward its next credit."""
        with self._lock:
            return self._carry.get(coldkey, 0)

    def carries(self) -> dict[str, int]:
        """Snapshot of every non-zero carried remainder (rao)."""
        with self._lock:
            return {k: v for k, v in self._carry.items() if v}

    def coldkey_for_hotkey(self, hotkey: str) -> str:
        coldkey = self._chain.hotkey_owner(hotkey)
        if coldkey is None:
            raise UnknownHotkeyError(f"hotkey {hotkey!r} not on metagraph")
        return coldkey

    def debit_for_commit(self, hotkey: str, round_id: str) -> str:
        """Debit 1 credit from the hotkey's coldkey; returns the coldkey."""
        coldkey = self.coldkey_for_hotkey(hotkey)
        with self._lock:
            if self._balances.get(coldkey, 0) < 1:
                raise InsufficientCreditError(
                    f"coldkey {coldkey!r} (hotkey {hotkey!r}) has no credit")

            def apply() -> None:
                self._balances[coldkey] -= 1
                self._append("debit", round_id=round_id, coldkey=coldkey, credits=-1,
                             hotkey=hotkey, note="debited at commit acceptance")

            self._transaction(apply)
        return coldkey

    def refund_credit(self, coldkey: str, round_id: str, note: str) -> None:
        """Infra-DQ refund: a credit, never TAO (§2.4)."""
        with self._lock:
            def apply() -> None:
                self._balances[coldkey] = self._balances.get(coldkey, 0) + 1
                self._append("refund", round_id=round_id, coldkey=coldkey, credits=1,
                             note=note)

            self._transaction(apply)

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def events_json(self) -> str:
        """Publishable ledger (dashboard §9.1 'credit ledger')."""
        return json.dumps([asdict(e) for e in self.events], indent=1)

    @staticmethod
    def replay(events: list[LedgerEvent]) -> dict[str, int]:
        """Reconstruct all balances from the event log alone (E6 audit)."""
        balances: dict[str, int] = {}
        for e in events:
            if e.coldkey is not None and e.credits:
                balances[e.coldkey] = balances.get(e.coldkey, 0) + e.credits
        return {k: v for k, v in balances.items() if v or k in balances}

    @staticmethod
    def replay_carry(events: list[LedgerEvent]) -> dict[str, int]:
        """Reconstruct every carried remainder from the event log alone."""
        carry: dict[str, int] = {}
        for e in events:
            if e.kind == "grant" and e.coldkey is not None:
                carry[e.coldkey] = e.carry_rao
        return {k: v for k, v in carry.items() if v}

    def _append(self, kind: str, *, round_id: str | None = None,
                coldkey: str | None = None, credits: int = 0, rao: int = 0,
                tx_id: str | None = None, hotkey: str | None = None,
                note: str = "", carry_rao: int = 0) -> None:
        self._events.append(LedgerEvent(
            seq=len(self._events), kind=kind, round_id=round_id,
            coldkey=coldkey, credits=credits, rao=rao, tx_id=tx_id,
            hotkey=hotkey, note=note, carry_rao=carry_rao))

    def _transaction(self, fn):
        """Apply ``fn`` then persist; roll memory back if either fails.

        Must be called with ``self._lock`` held.
        """
        saved = (list(self._events), dict(self._balances), dict(self._carry),
                 set(self._seen_tx), self._last_block, self._fee_rao,
                 dict(self._pinned_rounds))
        try:
            result = fn()
            self._persist()
            return result
        except Exception:
            (self._events, self._balances, self._carry, self._seen_tx,
             self._last_block, self._fee_rao, self._pinned_rounds) = saved
            raise

    def _chain_cursor(self) -> int | None:
        getter = getattr(self._chain, "scan_cursor", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception:
            return None

    def _restore_chain_cursor(self, cursor: int | None) -> None:
        setter = getattr(self._chain, "restore_scan_cursor", None)
        if cursor is None or setter is None:
            return
        try:
            setter(cursor)
        except Exception as e:
            log.warning("could not roll back payment scan cursor to %s: %s", cursor, e)

    def _persist_cursor_only(self, cursor_before: int | None) -> None:
        if self._store_path is None:
            return
        cursor_now = self._chain_cursor()
        if cursor_now is None or cursor_now == cursor_before:
            return
        with self._lock:
            try:
                self._persist()
            except Exception as e:
                log.warning("payment ledger cursor checkpoint failed: %s", e)

    def _snapshot(self) -> dict:
        return {
            "version": LEDGER_VERSION,
            "updated_at": int(time.time()),
            "fee_rao": self._fee_rao,
            "last_block": self._last_block,
            "scan_cursor": self._chain_cursor(),
            "pinned_rounds": dict(self._pinned_rounds),
            "balances": {k: v for k, v in self._balances.items() if v},
            "carry_rao": {k: v for k, v in self._carry.items() if v},
            "seen_tx": sorted(self._seen_tx),
            "events": [asdict(e) for e in self._events],
        }

    def _persist(self) -> None:
        if self._store_path is None:
            return
        atomic_write_json(self._store_path, self._snapshot())

    def _load(self) -> None:
        assert self._store_path is not None
        data = load_ledger(self._store_path)
        if data is None:
            log.info("payment ledger %s absent; starting a new ledger", self._store_path)
            return
        events: list[LedgerEvent] = []
        for i, raw in enumerate(data["events"]):
            try:
                ev = LedgerEvent(
                    seq=int(raw["seq"]), kind=str(raw["kind"]),
                    round_id=raw.get("round_id"), coldkey=raw.get("coldkey"),
                    credits=int(raw.get("credits") or 0), rao=int(raw.get("rao") or 0),
                    tx_id=raw.get("tx_id"), hotkey=raw.get("hotkey"),
                    note=str(raw.get("note") or ""),
                    carry_rao=int(raw.get("carry_rao") or 0))
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                raise LedgerStoreError(
                    f"payment ledger {self._store_path}: malformed event #{i}: {e}") from e
            if ev.seq != i:
                raise LedgerStoreError(
                    f"payment ledger {self._store_path}: event #{i} has seq {ev.seq}")
            events.append(ev)
        replayed = {k: v for k, v in self.replay(events).items() if v}
        try:
            stored = {str(k): int(v) for k, v in (data.get("balances") or {}).items() if int(v)}
        except (TypeError, ValueError) as e:
            raise LedgerStoreError(f"payment ledger {self._store_path}: bad balances: {e}") from e
        if stored != replayed:
            raise LedgerStoreError(
                f"payment ledger {self._store_path}: stored balances do not match "
                f"the replayed event log (stored={stored}, replayed={replayed})")
        replayed_carry = self.replay_carry(events)
        if "carry_rao" in data:
            try:
                stored_carry = {str(k): int(v) for k, v in (data["carry_rao"] or {}).items()
                                if int(v)}
            except (TypeError, ValueError) as e:
                raise LedgerStoreError(
                    f"payment ledger {self._store_path}: bad carry_rao: {e}") from e
            if stored_carry != replayed_carry:
                raise LedgerStoreError(
                    f"payment ledger {self._store_path}: stored carried remainders do "
                    f"not match the replayed event log (stored={stored_carry}, "
                    f"replayed={replayed_carry})")
        seen = {str(tx) for tx in (data.get("seen_tx") or [])}
        seen.update(e.tx_id for e in events if e.kind == "grant" and e.tx_id)
        pinned = {str(k): int(v) for k, v in (data.get("pinned_rounds") or {}).items()}
        for e in events:
            if e.kind == "fee_pin" and e.round_id:
                pinned[e.round_id] = e.rao
        fee = data.get("fee_rao")
        if fee is None:
            pins = [e.rao for e in events if e.kind == "fee_pin" and e.rao > 0]
            fee = pins[-1] if pins else None
        self._events = events
        self._balances = replayed
        self._carry = replayed_carry
        self._seen_tx = seen
        self._pinned_rounds = pinned
        self._fee_rao = int(fee) if fee is not None else None
        try:
            self._last_block = int(data.get("last_block", -1))
        except (TypeError, ValueError):
            self._last_block = max((e_block for e_block in
                                    (self._block_of(e.tx_id) for e in events if e.kind == "grant")
                                    if e_block is not None), default=-1)
        cursor = data.get("scan_cursor")
        if cursor is not None:
            self._restore_chain_cursor(int(cursor))
        log.info("payment ledger %s loaded: %d event(s), %d funded coldkey(s), "
                 "%d processed transfer(s), last_block=%d, scan_cursor=%s",
                 self._store_path, len(events), len(replayed), len(seen),
                 self._last_block, cursor)

    @staticmethod
    def _block_of(tx_id: str | None) -> int | None:
        """BittensorChainView tx ids are ``"<block>-<idx>"``; MockChain's are opaque."""
        if not tx_id or "-" not in tx_id:
            return None
        head = tx_id.split("-", 1)[0]
        return int(head) if head.isdigit() else None
