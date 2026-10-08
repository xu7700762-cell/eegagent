# Cloud judgement and improved numerical candidates

`improve.py` adds path MIL, regional/channel spectral candidates, training-reference tangent covariance and an internally selected numerical procedure. `prepare.py` rebuilds label-free evidence from frozen pilot models. `cloud.py` handles Responses API requests, blind prompts, parsing, retries and logs. `supervisor.py` provides final judgement/fusion.

```python
from vrms_cloud.supervisor import assess_evidence
result = assess_evidence(evidence, internal_examples, numeric_probability, log_path, llm_weight=.25)
```

Configure `EEG_GPT_BASE_URL`, `EEG_GPT_MODEL`, `EEG_GPT_KEY` in the environment or an explicit project `.env`. It never reads another application's credentials. The Responses model must support the supplied schema and reasoning parameters.

```bash
python -m vrms_cloud.prepare
python -m vrms_cloud.improve
# Actual API calls:
python -m vrms_cloud.cloud --modes zero_shot few_shot improved_few_shot --workers 3
python -m vrms_cloud.independent --workers 4
# Cached evaluation and frozen-model inference checks:
python -m vrms_cloud.evaluate
python -m vrms_cloud.validate
```

Default outputs: `outputs/vrms_cloud/20261008_seed2026`. Preparation/improvement overwrite stage artifacts. Independent mode contains one held-out path and internal examples. Fold batches remain a separate diagnostic context. Failures keep the numerical result; confidence is uncalibrated. See [reproduction](../docs/REPRODUCTION.md) and [protocol](../docs/PROTOCOL.md).
