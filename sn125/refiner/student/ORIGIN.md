# Pinned student optimizer for the Refiner task

`kairos_b_96fe905a.py` is a byte-for-byte copy of the Challenge-1 (optimizer task)
submission whose code hash is

    sha256 96fe905a538c32325bf550f0eed66e9ed5289489589752bbe0c8da81771ea383

("KAIROS-B": shape-bucketed Muon on hidden matrices, Adam on embeddings/norms,
fp32 master weights with bf16 error feedback, 1-sqrt cooldown). It was the
production frontier on FW2_smol_360M at the time the Refiner task was built
(round records of September 2026; the operator's review of that submission is
internal).

On the Refiner task this optimizer is TRUSTED HARNESS CODE, not a submission:
both the identity baseline and every refiner submission train the student with
exactly this update rule, so the measured difference is attributable to the
refiner alone. Its sha256 is recorded in every result JSON
(`student_optimizer.sha256`). Changing the pin changes the task.

Override for experiments: `SN125_STUDENT_OPTIMIZER_PATH=/path/to/optimizer.py`.
If the pinned file is missing (e.g. a trimmed public checkout), the harness
falls back to `sn125.references.ADAMW_SOURCE` and says so in the result.
