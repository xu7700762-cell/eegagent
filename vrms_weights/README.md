# Cached probability weight scan

`scan.py` reads saved independent GPT/DeepSeek scores. No API or training is performed. Fusion is `(1-w)*numeric + w*cloud`, threshold `>=.5`. It checks the1% grid and every exact crossing/open interval, including isolated threshold-equality maxima.

```bash
python -m unittest vrms_weights.test_scan -v
python -m vrms_weights.scan
```

Default outputs: `outputs/vrms_weights/20261008_seed2026`. Existing output directories are refused; use a new `--out` to preserve scans. Maxima select weights on already examined outer labels. They are post hoc diagnostics, not validated deployment weights or search-corrected significance results.

The fold-policy diagnostic uses old GPT batch validation and matching batch test predictions, separately from independent requests. Later refinement has separate independent policy predictions; this module does not mix them. Original logs, source hashes, labels and probabilities are checked. See [reproduction](../docs/REPRODUCTION.md).
