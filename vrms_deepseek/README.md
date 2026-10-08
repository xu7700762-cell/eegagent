# Standalone DeepSeek Supervisor

This release includes an independent Chat Completions adapter in `provider.py`. It preserves JSON-object mode, temperature0,700-token output limit,60-second timeout, SDK retries disabled, at most two wrapper attempts and a shared call budget. Native DeepSeek flash/pro disables thinking.

Configure `EEG_API_BASE_URL`, `EEG_API_MODEL`, `EEG_API_KEY`. Environment values override the optional project `.env`; `--config-file` selects a file. No other local EEG project or application login is needed.

```bash
# Requires pilot/cloud artifacts; actual API calls:
python -m vrms_deepseek.experiment --workers 4
# Probe only:
python -m vrms_deepseek.experiment --limit 1
# Cached audit and evaluation:
python -m vrms_deepseek.evaluate
python -m unittest vrms_deepseek.test_contracts -v
```

Default outputs: `outputs/vrms_deepseek/20261008_seed2026`. Evidence and examples match the independent GPT branch; numerical models/splits are frozen and the comparison weight is25%. Failures preserve numeric fallback. Logs exclude credentials and keep model identity, attempts, usage and latency. Adapter packaging differs from the historical local client; a changed source requires a new protocol and hashes.
