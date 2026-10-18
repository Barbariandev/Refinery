"""Downstream benchmark suite for the Refiner task (placeholder: easy, old, public).

The optimizer task is scored on held-out FineWeb-Edu loss. The Refiner task
cannot be: the refiner chooses what the student learns from, so a loss on the
same distribution would reward teaching-to-the-test rather than learning. The
Refiner task is scored on a pre-tokenized suite of small-model benchmarks
(HellaSwag, PIQA, ARC-Easy, SciQ, OpenBookQA, ...; `suite.py`) with
likelihood-based metrics (`score.py`). Held-out loss is still reported, as a
diagnostic only.
"""
