"""SN125 payments — TAO payment gate (DESIGN.md §8).

registry : prepaid-evaluation credit ledger behind a thin chain interface,
           thread-safe and durably persisted (store.py) across restarts.
chain    : read-only bittensor ChainView with incremental scan cursor and
           archive-node backfill for blocks a lite node has pruned.
watch    : background scanner that keeps the ledger current for the whole
           validator lifetime, not just at round start.
"""
