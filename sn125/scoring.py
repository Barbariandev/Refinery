"""SN125 — Scoring: score_submission + generalization gap."""
import math
import statistics
from .training import TrainingCurve, ScoreRecord


def significance_pareto_adjustment(
    per_task_pairs: dict[str, list[tuple[float, float]]],
    task_weights: dict[str, float],
    z: float = 1.0,
    min_effect: float = 0.0,
    regression_cap: float = 0.5,
    unproven_haircut: float = 0.5,
) -> dict:
    """Paired significance + Pareto-dominance discipline over per-task seed trials.

    Each task carries a list of seed-paired ``(sub_final_loss, base_final_loss)``
    measurements — the SAME seed is used for the submission and its baseline, so
    the per-seed relative improvement ``r_i = (base_i - sub_i) / base_i`` is a
    paired sample. From it we estimate one-sided confidence bounds on the true
    improvement:

        mean = mean(r_i)
        se   = stdev(r_i, sample) / sqrt(n)
        lcb  = mean - z*se        # lower bound
        ucb  = mean + z*se        # upper bound

    (Normal approximation. With very small ``n`` a larger ``z`` is more
    conservative; exact t-quantiles would be stricter still — documented, not
    applied, to avoid pulling in scipy.)

    Per task:
      - significant WIN : lcb >  min_effect   (confidently better than baseline)
      - significant LOSS: ucb < -min_effect   (confidently worse than baseline)
      - otherwise       : within seed noise — no credit either way.

    Pareto dominance: the submission dominates the baseline only if it has NO
    significant loss on any task AND at least one significant win. A single
    confident per-task regression breaks dominance.

    Returns diagnostics plus two scalars the caller applies to the final score,
    both of which can only REDUCE it (the discipline never grants credit):
      - ``regression_penalty``: weighted magnitude of significant per-task
        losses, capped at ``regression_cap``. Subtract from the final score.
      - ``haircut``: multiplicative factor in (0, 1]. Equals ``unproven_haircut``
        when there is no statistically significant win anywhere (the positive
        score is unproven over seed noise), else 1.0. Apply to positive scores.

    Tasks with fewer than 2 paired measurements are skipped (variance is
    unestimable); if none are evaluable the result is a neutral no-op.
    """
    per_task = {}
    evaluable = []
    for tid, pairs in per_task_pairs.items():
        rels = [(b - s) / b for (s, b) in pairs if b not in (0, 0.0)]
        n = len(rels)
        if n < 2:
            continue
        mean = statistics.fmean(rels)
        se = statistics.stdev(rels) / math.sqrt(n)
        lcb = mean - z * se
        ucb = mean + z * se
        is_win = lcb > min_effect
        is_loss = ucb < -min_effect
        per_task[tid] = {
            "n": n, "mean": round(mean, 6), "se": round(se, 6),
            "lcb": round(lcb, 6), "ucb": round(ucb, 6),
            "significant_win": is_win, "significant_loss": is_loss,
        }
        evaluable.append(tid)

    if not evaluable:
        return {"evaluable": False, "regression_penalty": 0.0, "haircut": 1.0,
                "pareto_dominates": None, "sig_wins": [], "sig_losses": [],
                "per_task": per_task}

    sig_wins = [t for t in evaluable if per_task[t]["significant_win"]]
    sig_losses = [t for t in evaluable if per_task[t]["significant_loss"]]

    total_w = sum(task_weights.get(t, 1.0) for t in evaluable) or 1.0
    regression_penalty = sum(
        (task_weights.get(t, 1.0) / total_w) * abs(min(0.0, per_task[t]["ucb"]))
        for t in sig_losses)
    regression_penalty = min(regression_cap, regression_penalty)

    haircut = unproven_haircut if not sig_wins else 1.0
    pareto_dominates = (not sig_losses) and bool(sig_wins)

    def _gated(t):
        d = per_task[t]
        if d["significant_win"]:
            return d["lcb"]
        if d["significant_loss"]:
            return d["ucb"]
        return 0.0
    gated_cross_scale = min(_gated(t) for t in evaluable)

    return {
        "evaluable": True,
        "regression_penalty": round(regression_penalty, 6),
        "haircut": haircut,
        "pareto_dominates": pareto_dominates,
        "sig_wins": sig_wins, "sig_losses": sig_losses,
        "gated_cross_scale": round(gated_cross_scale, 6),
        "z": z, "min_effect": min_effect,
        "per_task": per_task,
    }


def _generalization_gap(eval_curve: list[tuple[int, float]],
                        train_curve: list[tuple[int, float]]) -> float:
    """Measure overfitting: how much worse eval is than train at end of training.
    Returns (eval_avg - train_avg) / train_avg using last 3 points for stability.
    Positive = overfitting, 0 = perfect generalization, negative = eval better than train.
    Uses average of last min(3, len) points to reduce noise from single-point measurement.
    """
    if not eval_curve or not train_curve:
        return 0.0
    n = min(3, len(eval_curve), len(train_curve))
    eval_avg = sum(p[1] for p in eval_curve[-n:]) / n
    train_avg = sum(p[1] for p in train_curve[-n:]) / n
    if abs(train_avg) < 1e-9:
        return 0.0
    return (eval_avg - train_avg) / abs(train_avg)


def score_submission(
    submission_curves: dict[str, TrainingCurve],
    baseline_curves: dict[str, TrainingCurve],
    task_weights: dict[str, float],
    source_bytes: int,
    max_source_bytes: int = 1_048_576,
    complexity_lambda: float = 0.002,
    memory_beta: float = 0.01,
) -> ScoreRecord:
    """Score a submission against baselines. Fixed-budget (fixed-FLOPs) scoring.

    Loss-at-fixed-budget: every submission gets exactly one full-horizon run on the
    same 20h / ~$100 b200-small budget (N≈762,828 steps, ≈Chinchilla-optimal). The
    objective is purely the *relative held-out-loss improvement* over the rolling-best
    baseline at that fixed budget. Slower-per-step optimizers simply get fewer steps
    in the 20h wall (enforced by compute_budget_seconds in train_and_eval), so their
    final loss already reflects quality-at-fixed-compute.

    Per SPEC §7, the budget itself prices slowness and heaviness, so the auxiliary
    penalties are deliberately *near-zero*, not load-bearing:
      - No multiplicative efficiency factor — obsolete under a fixed budget.
      - complexity_lambda=0.002 → at most a 0.002 tie-breaker (source_bytes/50k);
        kept tiny purely to deter gratuitously bloated source, not to shape ranking.
      - memory_beta=0.01 → a small *relative* state-size nudge (2× heavier than the
        baseline costs ~0.01 score); the 20h wall already penalizes heavy state via
        fewer steps. Set either to 0.0 to disable entirely.

    Significance at K=1 eval/round: per-round seed-paired significance is unavailable
    (one full-budget run per submission). Confidence comes instead from (a) the
    rolling-best bar, which moves only on a genuine beat-by-τ, and (b) cross-round
    confirmation — a new king must clear the bar and is re-verified before reward
    fully vests (DESIGN §7.4). This function returns the single-round score; the
    cross-round vesting/confirmation policy lives in the reward/reign logic.

    3 components (conv=0.4, cross=0.4, gen=0.2) + memory penalty:
      - Convergence (final loss): weighted avg improvement across tasks
      - Cross-scale: min(per-task convergence) — robustness across model sizes
      - Generalization: overfits less than baseline? (small additive bonus, ≤0.2)
      - Memory: -β × log2(state_multiplier) penalizes heavy optimizer state
        β=0.01: 4× state loses ~0.01 vs 2× state (~10-20% of typical good scores)
    """
    task_details = {}
    best_hparams = {}
    failed = []

    for task_id, sub_c in submission_curves.items():
        base_c = baseline_curves.get(task_id)

        if sub_c.failed or not sub_c.eval_points or base_c is None or base_c.failed or not base_c.eval_points:
            failed.append(task_id)
            task_details[task_id] = {"convergence": 0.0, "gen": 0.0}
            continue

        best_hparams[task_id] = (sub_c.lr, sub_c.wd)

        base_final = base_c.eval_points[-1][1]
        sub_final = sub_c.eval_points[-1][1]
        conv_imp = (base_final - sub_final) / max(abs(base_final), 1e-9)
        conv_imp = max(-2.0, min(1.0, conv_imp))

        sub_gap = _generalization_gap(sub_c.eval_points, sub_c.train_points)
        base_gap = _generalization_gap(base_c.eval_points, base_c.train_points)
        gen_imp = (base_gap - sub_gap) / max(abs(base_gap) + 0.01, 1e-9)
        gen_imp = max(-1.0, min(1.0, gen_imp))

        task_details[task_id] = {
            "convergence": conv_imp, "gen": gen_imp,
            "sub_final_loss": sub_final, "base_final_loss": base_final,
            "sub_time": sub_c.wall_seconds, "base_time": base_c.wall_seconds,
            "sub_steps": sub_c.eval_points[-1][0] if sub_c.eval_points else 0,
            "base_steps": base_c.eval_points[-1][0] if base_c.eval_points else 0,
            "sub_gap": sub_gap, "base_gap": base_gap,
        }

    _missing_w = [t for t in task_details if t not in failed and t not in task_weights]
    if _missing_w:
        raise ValueError(f"task_weights missing keys: {_missing_w}")
    total_w = sum(task_weights[t] for t in task_details if t not in failed)
    if total_w < 1e-9:
        return ScoreRecord(-1.0, {"convergence": 0, "generalization": 0,
            "cross_scale": -1.0,
            "complexity_penalty": complexity_lambda * (source_bytes / max_source_bytes)},
            task_details, {}, failed)

    agg = {"conv": 0.0, "gen": 0.0}
    task_convs = []
    for tid, d in task_details.items():
        if tid in failed:
            task_convs.append(-1.0)
            continue
        tw = task_weights[tid] / total_w
        agg["conv"] += d["convergence"] * tw
        agg["gen"] += d["gen"] * tw
        task_convs.append(d["convergence"])

    cross = min(task_convs) if task_convs else 0.0
    complexity = complexity_lambda * (source_bytes / max_source_bytes)

    _state_mults = [c.state_multiplier for c in submission_curves.values()
                    if not c.failed and c.state_multiplier > 0]
    _base_mults = [c.state_multiplier for c in baseline_curves.values()
                   if not c.failed and c.state_multiplier > 0]
    max_state_mult = max(max(_state_mults, default=1.0), 1.0)
    base_state_mult = max(max(_base_mults, default=1.0), 1.0)
    memory_penalty = memory_beta * math.log2(max_state_mult / base_state_mult)

    quality = 0.4 * agg["conv"] + 0.4 * cross + 0.2 * agg["gen"]
    final = quality - complexity - memory_penalty

    components = {
        "convergence": agg["conv"],
        "generalization": agg["gen"],
        "cross_scale": cross,
        "complexity_penalty": complexity,
        "memory_score": {"state_multiplier": round(max_state_mult, 3),
                         "base_state_multiplier": round(base_state_mult, 3),
                         "penalty": round(memory_penalty, 6)},
    }

    return ScoreRecord(
        final_score=final, components=components,
        task_scores=task_details,
        best_hparams={k: list(v) for k, v in best_hparams.items()},
        failed_tasks=failed,
    )
