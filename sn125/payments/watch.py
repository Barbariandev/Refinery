"""Continuous treasury-transfer scanner (background thread).

The round loop spends ~20 h inside the evaluation phase and then sleeps to the
24 h boundary. Syncing the payment registry only at round start therefore
left every deposit that landed during a round unscanned until the next one —
and, on a lite node that prunes old block events, potentially unscannable.

``start_payment_watcher`` runs ``registry.sync_to_head()`` on a short cadence
(well inside the lite node's retention) for the whole validator lifetime, so
a deposit becomes a credit within about ``interval_s + finality`` regardless
of where the round loop is. The registry is thread-safe and persists every
grant, so the watcher can run alongside commit debits on the main thread.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from .registry import FeeNotPinnedError, PaymentRegistry

log = logging.getLogger("sn125.payments.watch")

DEFAULT_SCAN_INTERVAL_S = 60.0
SCAN_INTERVAL_ENV = "SN125_PAYMENT_SCAN_S"


def start_payment_watcher(
    registry: PaymentRegistry,
    *,
    interval_s: float = DEFAULT_SCAN_INTERVAL_S,
    stop: threading.Event | None = None,
    on_sync: Callable[[int], None] | None = None,
    name: str = "sn125-payment-watch",
) -> threading.Event:
    """Scan treasury transfers every ``interval_s`` seconds until the returned
    event is set. The first scan runs immediately.

    Errors never escape the thread: a failed scan is logged and retried on the
    next tick (the registry rolls back partial state itself). ``on_sync`` is
    invoked with the number of transfers processed after each scan.
    """
    if interval_s <= 0:
        raise ValueError("payment scan interval must be positive")
    stop = stop or threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                n = registry.sync_to_head()
                if n:
                    log.info("payment watcher credited %d treasury transfer(s)", n)
                if on_sync is not None:
                    on_sync(n)
            except FeeNotPinnedError:
                log.debug("payment watcher idle: no fee in force yet")
            except Exception as e:
                log.warning("payment watcher scan failed (retry in %.0fs): %s",
                            interval_s, e)
            if stop.wait(interval_s):
                break

    threading.Thread(target=loop, daemon=True, name=name).start()
    return stop
