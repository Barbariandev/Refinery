"""Real bittensor ``ChainView`` for :class:`PaymentRegistry` (registry.py / §5.3).

The registry needs exactly two read-only chain facts (registry.ChainView):

  * ``finalized_transfers_to_treasury(since_block)`` — finalized TAO transfers
    into the treasury coldkey, ascending by block, strictly above ``since_block``.
  * ``hotkey_owner(hotkey)`` — the coldkey owning a hotkey on the metagraph.

Both are **read-only**: this watcher holds no signing keys and never submits an
extrinsic (the §5.3 no-keys property the registry docstring promises). The only
config surface is the treasury **address** + finality depth — both public.

Design points:

* **Finality.** A transfer is only credited once its block is at least
  ``finality_depth`` below the chain head, so a reorg can never strand a credit.
* **Incremental scan cursor.** The view keeps a ``_scanned_through`` watermark
  and scans only *new* finalized blocks each call (never a deep re-scan). The
  registry persists that cursor with the ledger (``scan_cursor`` /
  ``restore_scan_cursor``) so a restart resumes exactly where it stopped.
* **Prune window + archive backfill.** A finney lite node discards the state
  (and therefore the events) of old blocks — measured 2026-09-08 at roughly
  3 000 blocks (~10 h). Blocks older than ``max_scan_blocks`` below the
  finalized head are read from the optional ``archive_events`` reader (an
  archive node) instead; without one they are skipped with a loud warning, as
  the deposits in them cannot be recovered from a lite node. This is what
  makes a start-up **backfill** (``start_block``) and a recovery after a long
  outage possible.
* **Chunking.** One call scans at most ``chunk_blocks`` blocks so a large
  backfill is persisted incrementally by the registry; callers loop until
  ``caught_up()`` (``PaymentRegistry.sync_to_head``).

To keep this module import-light and unit-testable on CPU with no live chain (the
adapter.py pattern), the chain primitives are **injected callables**. The
``from_subtensor`` classmethod binds them to a live ``bittensor.Subtensor``;
tests bind fakes. Nothing here imports bittensor.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from .registry import Transfer

log = logging.getLogger("sn125.payments.chain")

DEFAULT_MAX_SCAN_BLOCKS = 2000
DEFAULT_FINALITY_DEPTH = 10
DEFAULT_CHUNK_BLOCKS = 500
_PROGRESS_EVERY_BLOCKS = 500
BLOCK_SECONDS = 12.0
DEFAULT_ARCHIVE_NETWORK = "archive"


def _looks_discarded(exc: BaseException) -> bool:
    """True for the lite node's 'state discarded / block too old' failures."""
    text = f"{type(exc).__name__} {exc}".lower()
    return ("statediscarded" in text or "state discarded" in text
            or "too old" in text or "unknown block" in text)


def blocks_for_hours(hours: float) -> int:
    return max(0, int(float(hours) * 3600.0 / BLOCK_SECONDS))


class BittensorChainView:
    """Read-only ChainView over injected chain primitives.

    Parameters
    ----------
    treasury_coldkey:
        SS58 of the coldkey that receives evaluation fees. Only transfers whose
        *destination* is this address are surfaced.
    get_head:
        ``() -> int`` current chain head block number.
    get_block_hash:
        ``(block:int) -> str`` block hash for a finalized block.
    get_events:
        ``(block_hash:str) -> list`` raw substrate events for that block.
    get_owner:
        ``(hotkey:str) -> str | None`` coldkey owning a hotkey, or None.
    archive_events:
        optional ``(block:int) -> list`` events reader backed by an archive
        node, used for blocks the primary endpoint has pruned.
    start_block:
        first block to scan when no cursor has been restored (backfill start).
        None = start at the prune window (legacy behaviour).
    finality_depth, max_scan_blocks, chunk_blocks:
        see module docstring.
    """

    def __init__(
        self,
        treasury_coldkey: str,
        *,
        get_head: Callable[[], int],
        get_block_hash: Callable[[int], str],
        get_events: Callable[[str], list],
        get_owner: Callable[[str], Optional[str]],
        archive_events: Callable[[int], list] | None = None,
        start_block: int | None = None,
        finality_depth: int = DEFAULT_FINALITY_DEPTH,
        max_scan_blocks: int = DEFAULT_MAX_SCAN_BLOCKS,
        chunk_blocks: int = DEFAULT_CHUNK_BLOCKS,
    ) -> None:
        if not treasury_coldkey:
            raise ValueError("treasury_coldkey is required")
        if finality_depth < 0:
            raise ValueError("finality_depth must be >= 0")
        if max_scan_blocks < 1:
            raise ValueError("max_scan_blocks must be >= 1")
        if chunk_blocks < 1:
            raise ValueError("chunk_blocks must be >= 1")
        self.treasury_coldkey = treasury_coldkey
        self._get_head = get_head
        self._get_block_hash = get_block_hash
        self._get_events = get_events
        self._get_owner = get_owner
        self._archive_events = archive_events
        self.finality_depth = finality_depth
        self.max_scan_blocks = max_scan_blocks
        self.chunk_blocks = chunk_blocks
        self._scanned_through = -1
        self._explicit_floor = start_block is not None
        if start_block is not None:
            self._scanned_through = max(-1, int(start_block) - 1)
        self._archive_used = 0

    def scan_cursor(self) -> int:
        """Highest finalized block this view has fully scanned (-1 = none)."""
        return self._scanned_through

    def restore_scan_cursor(self, block: int) -> None:
        """Resume scanning strictly after ``block`` (restart recovery / rewind)."""
        self._scanned_through = max(-1, int(block))
        self._explicit_floor = True

    def has_archive(self) -> bool:
        return self._archive_events is not None

    def finalized_head(self) -> int:
        return int(self._get_head()) - self.finality_depth

    def caught_up(self) -> bool:
        """True when every finalized block has been scanned."""
        return self._scanned_through >= self.finalized_head()

    def hotkey_owner(self, hotkey: str) -> Optional[str]:
        return self._get_owner(hotkey)

    def finalized_transfers_to_treasury(self, since_block: int) -> list[Transfer]:
        finalized_head = self.finalized_head()
        if finalized_head <= since_block and finalized_head <= self._scanned_through:
            return []

        floor = max(since_block, self._scanned_through)
        window_floor = finalized_head - self.max_scan_blocks
        if floor < window_floor:
            if self._archive_events is not None and not self._explicit_floor:
                log.info("treasury transfer scan: no backfill start configured; "
                         "starting at the prune window (block %d)", window_floor + 1)
                floor = window_floor
            elif self._archive_events is None:
                log.warning(
                    "treasury transfer scan gap: blocks (%d, %d] are below the "
                    "%d-block prune window and were not scanned (validator offline "
                    "too long? configure an archive endpoint to backfill)",
                    floor, window_floor, self.max_scan_blocks,
                )
                floor = window_floor
            else:
                log.info(
                    "treasury transfer scan: blocks (%d, %d] are past the lite-node "
                    "prune window; reading them from the archive endpoint",
                    floor, min(window_floor, floor + self.chunk_blocks),
                )
        start = floor + 1
        end = min(finalized_head, start + self.chunk_blocks - 1)
        if end < start:
            return []
        total = end - start + 1
        if total >= _PROGRESS_EVERY_BLOCKS:
            log.info("treasury transfer scan: blocks %d..%d (%d blocks, finalized head %d)",
                     start, end, total, finalized_head)

        out: list[Transfer] = []
        t0 = time.time()
        for block in range(start, end + 1):
            try:
                events = self._read_events(block, block <= window_floor)
            except Exception as e:
                if _looks_discarded(e) and self._archive_events is None:
                    log.error("block %d is pruned on the primary endpoint and no "
                              "archive endpoint is configured; any treasury deposit "
                              "in it CANNOT be credited automatically (%s)", block, e)
                    self._scanned_through = max(self._scanned_through, block)
                    continue
                log.warning("failed reading events for block %d: %s", block, e)
                self._scanned_through = max(self._scanned_through, block - 1)
                out.sort(key=lambda t: (t.block, t.tx_id))
                return out
            for idx, ev in enumerate(events):
                parsed = self._transfer_to_treasury(ev, block, idx)
                if parsed is not None:
                    out.append(parsed)
            self._scanned_through = max(self._scanned_through, block)
            done = block - start + 1
            if done % _PROGRESS_EVERY_BLOCKS == 0 and done < total:
                log.info("treasury transfer scan: %d/%d blocks, %d transfer(s) so far "
                         "(%.0fs)", done, total, len(out), time.time() - t0)

        if total >= _PROGRESS_EVERY_BLOCKS:
            log.info("treasury transfer scan: blocks %d..%d done, %d treasury "
                     "transfer(s) in %.0fs%s", start, end, len(out), time.time() - t0,
                     "" if self.caught_up() else f"; {finalized_head - end} more to scan")
        out.sort(key=lambda t: (t.block, t.tx_id))
        return out

    def _read_events(self, block: int, past_window: bool) -> list:
        """Events of ``block``: archive when pruned/past the window, else primary."""
        if past_window and self._archive_events is not None:
            self._archive_used += 1
            return self._archive_events(block)
        try:
            return self._get_events(self._get_block_hash(block))
        except Exception as e:
            if self._archive_events is None or not _looks_discarded(e):
                raise
            log.info("block %d pruned on the primary endpoint; reading it from the "
                     "archive endpoint", block)
            self._archive_used += 1
            return self._archive_events(block)

    def _transfer_to_treasury(self, ev, block: int, idx: int) -> Optional[Transfer]:
        """Map one raw substrate event to a treasury ``Transfer`` or None.

        Robust to the two event shapes seen across substrate-interface versions:
        ``{'event': {'module_id','event_id','attributes'}}`` and the flatter
        ``{'module_id','event_id','attributes'}``; attributes may be a dict
        (``from``/``to``/``amount``) or a positional ``(from, to, amount)``.
        """
        module, name, attrs = _event_fields(ev)
        if module != "Balances" or name != "Transfer":
            return None
        src, dest, amount = _transfer_attrs(attrs)
        if dest != self.treasury_coldkey or src is None or amount is None:
            return None
        try:
            amount_rao = int(amount)
        except (TypeError, ValueError):
            return None
        if amount_rao <= 0:
            return None
        ex_idx = _maybe_int(_get(ev, "extrinsic_idx"))
        tag = ex_idx if ex_idx is not None else idx
        return Transfer(tx_id=f"{block}-{tag}", src_coldkey=str(src),
                        amount_rao=amount_rao, block=block)

    @classmethod
    def from_subtensor(cls, subtensor, treasury_coldkey: str, *,
                       archive_subtensor_factory: Callable[[], object] | None = None,
                       owner_subtensor=None,
                       **kw):
        """Bind to a live ``bittensor.Subtensor`` (`subtensor.substrate`).

        The substrate websocket is NOT thread-safe, so ``subtensor`` should be
        a connection dedicated to the payment scanner (which runs on its own
        thread), while ``owner_subtensor`` (default: ``subtensor``) serves the
        ``hotkey_owner`` lookups the round loop makes on the main thread at
        commit time — pass the validator's own connection there.

        ``archive_subtensor_factory`` (e.g. ``lambda: bt.Subtensor("archive")``)
        is invoked lazily, on the first block the primary endpoint cannot
        serve, and the connection is then reused by the scanner.
        """
        substrate = subtensor.substrate
        owner_subtensor = owner_subtensor or subtensor
        archive_events = kw.pop("archive_events", None)
        if archive_events is None and archive_subtensor_factory is not None:
            holder: dict = {}

            def archive_events(block: int) -> list:
                st = holder.get("subtensor")
                if st is None:
                    log.info("connecting to the archive endpoint for treasury backfill")
                    st = holder["subtensor"] = archive_subtensor_factory()
                sub = st.substrate
                return sub.get_events(sub.get_block_hash(block))

        return cls(
            treasury_coldkey,
            get_head=subtensor.get_current_block,
            get_block_hash=substrate.get_block_hash,
            get_events=substrate.get_events,
            get_owner=owner_subtensor.get_hotkey_owner,
            archive_events=archive_events,
            **kw,
        )



def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _maybe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _event_fields(ev):
    """Return (module_id, event_id, attributes) from any known event shape."""
    inner = _get(ev, "event")
    if inner is None:
        inner = ev
    module = _get(inner, "module_id") or _get(inner, "module")
    name = _get(inner, "event_id") or _get(inner, "event")
    attrs = _get(inner, "attributes")
    if attrs is None:
        attrs = _get(inner, "params")
    return module, name, attrs


def _transfer_attrs(attrs):
    """Return (src, dest, amount) from a Transfer event's attributes."""
    if isinstance(attrs, dict):
        return attrs.get("from"), attrs.get("to"), attrs.get("amount")
    if isinstance(attrs, (list, tuple)) and len(attrs) >= 3:
        vals = [a.get("value") if isinstance(a, dict) else a for a in attrs[:3]]
        return vals[0], vals[1], vals[2]
    return None, None, None
