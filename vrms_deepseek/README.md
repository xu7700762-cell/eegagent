# DeepSeek explanation-only V2

Uses the same frozen V2 models, calibrated probability, lazy ToolReplay and deterministic four-state Supervisor as the Responses branch. DeepSeek supplies evidence explanations only. Configuration is explicit project EEG_API_BASE_URL/MODEL/KEY, with no Codex credentials or provider fallback.

```bash
python -m vrms_deepseek.experiment --dry-run
# Actual API calls, followed by cached evaluation:
python -m vrms_deepseek.experiment --workers 4
python -m vrms_deepseek.evaluate
```

Default output: `outputs/vrms_deepseek/v2_seed2026`; --pilot-out selects frozen numerical artifacts. Missing preparation is generated from deep/QC snapshots without physiological tools. --dry-run requires no API config/request and --offline evaluates that run. GPT and DeepSeek outcomes keep exactly the same p_cal; failures do not change machine state. Logs preserve retry history and omit credentials. Do not compare refusal-subset accuracy directly with all-path accuracy.