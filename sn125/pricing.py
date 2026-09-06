"""Round-fee pricing helpers.

The TAO reference is deliberately a small, reviewable data file. An operator
can update it once per day without changing the payment ledger: the resulting
TAO fee is pinned for each round before commits open.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

DEFAULT_TAO_USD = 250.0
FEE_MARGIN = 0.10
PRICE_FILE = Path(__file__).with_name("tao_price.json")


def tao_usd_price() -> float:
    """Return the current operator-published TAO/USD reference price."""
    raw = os.environ.get("SN125_TAO_USD_PRICE", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
    else:
        try:
            value = float(json.loads(PRICE_FILE.read_text()).get("usd", 0.0))
        except (OSError, ValueError, TypeError, AttributeError):
            value = 0.0
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_TAO_USD
    return value


def fee_tao_for_cost(cost_usd: float, *, price_usd: float | None = None,
                     margin: float = FEE_MARGIN) -> float:
    """Convert an estimated run cost into TAO, rounding up to 0.001 TAO."""
    cost = float(cost_usd)
    price = tao_usd_price() if price_usd is None else float(price_usd)
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("estimated run cost must be positive")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("TAO/USD price must be positive")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("fee margin must be non-negative")
    return math.ceil((cost * (1.0 + margin) / price) * 1000.0) / 1000.0
