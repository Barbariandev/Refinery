"""Refiner task (Challenge 2): learn the most from each dataset ENTRY.

The miner submits a trained module (source + safetensors weights) that sits
between the fixed FineWeb-Edu entry stream and the fixed student model. For
each entry it decides how the student learns from it: how the entry's own
tokens are segmented, ordered, repeated and packed into training sequences,
and a per-token weight (0 = read as context, no gradient). It never authors
tokens, never sees the student's parameters, and runs inside the same
wall-clock budget. See docs/REFINER_TASK.md.

Package layout:
  entries.py  document index over the hash-pinned shards + seeded entry stream
  lessons.py  the Lesson contract, span-provenance materialization, identity packer
  refproc.py  isolated refiner process (sandboxed source, validated weights)
  feeder.py   harness-side feeder: window, signals, lesson queue -> (ids, weights)
  task.py     task constants, TaskSpec, student-optimizer pin
"""
