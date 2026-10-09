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

## Separate forced-binary tool Agent experiment

`tool_agent_v4.py` uses real Responses function calls and ToolReplay to acquire EEG evidence before GPT returns High/Low. Its learned correction tool executes eight numerical tools; no LLM probability is produced. The locked FEMBA+MIL baseline remains 89/146. Iteration4 achieves105/146 (71.92%), correcting32 and damaging16 paths. The same-tool numerical fusion achieves107/146 (73.29%); these results do not establish an extra accuracy benefit from GPT. Auxiliary tool fitting uses19 subjects versus14 in the original baseline, and development repeatedly examines this dataset.

```bash
# Actual API calls, using the project's configured provider:
python -m vrms_cloud.tool_agent_v4
python -m vrms_cloud.evaluate_tool_agent --out outputs/vrms_agent_loso/v7_gpt_r1_20261008
# Recount, frozen-artifact checks, and all146 numerical tool replays; no API:
python -m vrms_cloud.audit_tool_agent_v4
```

The original V2 explanation-only Supervisor remains independent. All iterations, full replies, failed requests, numerical controls and protocol limits are documented in [TOOL_AGENT_ITERATIONS](../docs/TOOL_AGENT_ITERATIONS.md). The [GPT coordination architecture and three-seed results](../docs/GPT_MULTI_AGENT.md) explain the GPT controller, specialist numerical tools and nested correction audit. Checkpoints and logs live under `outputs/vrms_agent_loso`; report scripts default to `reports/` and accept `--desktop`. CSP artifacts must serialize the importable `vrms_cloud.agent_tools_v2.FilterBankCSP`, using imported training functions rather than training that class through `__main__`.
