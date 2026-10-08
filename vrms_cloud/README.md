# Cloud explanation and deterministic Supervisor V2

`prepare.py` reconstructs only deep/temporal/QC snapshots from fresh V2 checkpoints and freezes artifact hashes. `independent.py` uses actual ToolReplay lazy decisions before optional API analysis. `cloud.py` provides a strict explanation-only Responses schema; `supervisor.py` preserves the calibrated model probability and machine reliability state. Reliable and poor-quality paths skip cloud calls.

```python
from vrms_cloud.supervisor import assess_evidence
result = assess_evidence(evidence, retrieved_examples, raw_deep_probability, log_path,
                         reliability=machine_reliability, decision=lazy_decision)
```

The response has supporting_evidence/conflicting_evidence/missing_evidence/explanation. It cannot return a new probability/class. p_cal is independently calibrated deep probability; invalid/missing calibration yields null and uncertainty. Old score caches or changed request hashes are rejected. Failures retain the same p_cal and machine gate.

```bash
python -m vrms_cloud.prepare
python -m vrms_cloud.independent --dry-run
python -m vrms_cloud.evaluate --offline
python -m vrms_cloud.validate --offline
# Actual optional explanation API requests:
python -m vrms_cloud.independent --workers 4
python -m vrms_cloud.evaluate
python -m vrms_cloud.validate
```

Default output: `outputs/vrms_cloud/v2_seed2026`. Custom pilot/output paths use prepare --pilot-out and --out, then the same --out for later commands. Preparation requires full V2 results; existing preparation is not overwritten. Frozen checkpoints, eligibility/caches, source files and labels are checked. Batch mode is a legacy diagnostic; V2 cloud CLI delegates to the independent runner.

Configure EEG_GPT_BASE_URL/MODEL/KEY in project .env/environment only. Dry-run uses no configuration/API. `improve.py` contains historical MIL/numerical comparisons and is excluded from V2 probability decisions. Contract tests and data eligibility are not new model performance evidence.