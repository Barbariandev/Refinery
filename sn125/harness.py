"""
SN125 Optimizer Discovery Subnet — Core Harness (re-export facade).

This module re-exports everything from the split sub-modules so that
existing `from sn125.harness import X` statements keep working.

Sub-modules:
  - references: optimizer source strings + extract_hparams
  - sandbox: AST validation, safe torch/triton proxies, optimizer loading
  - training: types, training loop, evaluation, isolation
  - scoring: score_submission
"""

# Re-export: references
from .references import (
    ADAMW_SOURCE, SGDM_SOURCE, ADAM_SOURCE, SGD_SOURCE,
    LION_SOURCE, SCHEDULE_FREE_SOURCE, PRODIGY_SOURCE,
    MUON_SOURCE, FUSED_ADAMW_SOURCE, REFERENCE_OPTIMIZERS,
    extract_hparams,
)

# Re-export: sandbox
from .sandbox import (
    ALLOWED_IMPORTS, ALLOWED_TRITON_SUBMODULES, ALLOWED_TORCH_SUBMODULES,
    FORBIDDEN_NAMES, SandboxViolation,
    validate_source, load_optimizer_sandboxed,
    _ALLOWED_DUNDER_ATTRS, _FORBIDDEN_FRAME_ATTRS,
)

# Re-export: training types + functions
from .training import (
    TaskSpec, TrainingCurve, ScoreRecord,
    compute_auc, train_and_eval,
    evaluate_submission, evaluate_submission_isolated,
    cleanup_data_cache,
    # internals used by the test suite / tooling
    _generate_data, _load_real_data, _build_model,
    _make_safe_env, _SANDBOX_SETUP_SCRIPT, _WORKER_SCRIPT,
    _has_sandbox, _has_cgroup_v2, _cgroup_create, _cgroup_add_pid, _cgroup_cleanup,
    os,
)

# Re-export: sandbox proxies/internals used by the test suite
from .sandbox import (
    _make_safe_torch, _make_safe_triton,
    check_torch_tamper, snapshot_torch_identities,
)

# Re-export: scoring
from .scoring import (
    score_submission, _generalization_gap, significance_pareto_adjustment,
)

# Re-export: FineWeb-Edu production data pipeline (sn125.fineweb)
from .fineweb import make_fineweb_task, SHARD_DATASET_NAME
