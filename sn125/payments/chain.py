"""Real bittensor ``ChainView`` for :class:`PaymentRegistry` (registry.py / §5.3).

The registry needs exactly two read-only chain facts (registry.ChainView):

  * ``finalized_transfers_to_treasury(since_block)`` — finalized TAO transfers
    into the treasury coldkey, ascending by block, strictly above ``since_block``.
  * ``hotkey_owner(hotkey)`` — the coldkey owning a hotkey on the metagraph.

Both are **read-only**: this watcher holds no signing keys and never submits an
extrinsic (the §5.3 no-keys property the registry docstring promises). The only
config surface is the treasury **address** + finality depth — both public.

Two design points carried from KB lessons:

* **Finality.** A transfer is only credited once its block is at least
  ``finality_depth`` below the chain head, so a reorg can never strand a credit.
* **Pruning + accumulation.** A free finney lite node prunes block events to a
  short window (~256 blocks / ~50 min). The watcher therefore keeps an internal
  ``_scanned_through`` watermark and scans only *new* finalized blocks each call
  (never a deep re-scan), and clamps the scan floor to ``max_scan_blocks`` below
  the finalized head. If the validator was offline longer than that window the
  intervening blocks are unrecoverable from a lite node — that case is *logged*,
  not silently swallowed (an archive endpoint would be needed to backfill).

To keep this module import-light and unit-testable on CPU with no live chain (the
adapter.py pattern), the chain primitives are **injected callables**. The
``from_subtensor`` classmethod binds them to a live ``bittensor.Subtensor``;
tests bind fakes. Nothing here imports bittensor.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .registry import Transfer

log = logging.getLogger(__name__)

DEFAULT_MAX_SCAN_BLOCKS = 200
DEFAULT_FINALITY_DEPTH = 10


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
    finality_depth, max_scan_blocks:
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
        finality_depth: int = DEFAULT_FINALITY_DEPTH,
        max_scan_blocks: int = DEFAULT_MAX_SCAN_BLOCKS,
    ) -> None:
        if not treasury_coldkey:
            raise ValueError("treasury_coldkey is required")
        if finality_depth < 0:
            raise ValueError("finality_depth must be >= 0")
        if max_scan_blocks < 1:
            raise ValueError("max_scan_blocks must be >= 1")
        self.treasury_coldkey = treasury_coldkey
        self._get_head = get_head
        self._get_block_hash = get_block_hash
        self._get_events = get_events
        self._get_owner = get_owner
        self.finality_depth = finality_depth
        self.max_scan_blocks = max_scan_blocks
        self._scanned_through = -1

    def hotkey_owner(self, hotkey: str) -> Optional[str]:
        return self._get_owner(hotkey)

    def finalized_transfers_to_treasury(self, since_block: int) -> list[Transfer]:
        finalized_head = self._get_head() - self.finality_depth
        if finalized_head <= since_block and finalized_head <= self._scanned_through:
            return []

        floor = max(since_block, self._scanned_through)
        window_floor = finalized_head - self.max_scan_blocks
        if floor < window_floor:
            log.warning(
                "treasury transfer scan gap: blocks (%d, %d] are below the "
                "%d-block prune window and were not scanned (validator offline "
                "too long? archive endpoint needed to backfill)",
                floor, window_floor, self.max_scan_blocks,
            )
            floor = window_floor
        start = floor + 1

        out: list[Transfer] = []
        for block in range(start, finalized_head + 1):
            try:
                block_hash = self._get_block_hash(block)
                events = self._get_events(block_hash)
            except Exception as e:
                log.warning("failed reading events for block %d: %s", block, e)
                continue
            for idx, ev in enumerate(events):
                parsed = self._transfer_to_treasury(ev, block, idx)
                if parsed is not None:
                    out.append(parsed)

        self._scanned_through = max(self._scanned_through, finalized_head)
        out.sort(key=lambda t: (t.block, t.tx_id))
        return out

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
    def from_subtensor(cls, subtensor, treasury_coldkey: str, **kw):
        """Bind to a live ``bittensor.Subtensor`` (`subtensor.substrate`)."""
        substrate = subtensor.substrate
        return cls(
            treasury_coldkey,
            get_head=subtensor.get_current_block,
            get_block_hash=substrate.get_block_hash,
            get_events=substrate.get_events,
            get_owner=subtensor.get_hotkey_owner,
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
