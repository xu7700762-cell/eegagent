# Numerical pilot

`data.py` implements stateful causal CDT preprocessing, QC, spectral/covariance features and label-separated caches. `model.py` defines a from-scratch compact CNN. `engine.py` implements temporal evidence, support/conflict and lazy adaptive exits. `experiment.py` performs frozen24-subject LOSO with14/5/4/1 internal/test roles.

```bash
python -m vrms_pilot.experiment --stage prepare --data-root /path/to/data
python -m vrms_pilot.experiment --stage smoke
python -m vrms_pilot.experiment --stage run
python -m vrms_pilot.validate_results
python -m vrms_pilot.export_window_oof
```

Set `EEG_DATA_ROOT` to the same data root for source validation. Default outputs: `outputs/vrms_pilot/20261007_seed2026`. Preparation overwrites caches; preserve completed runs. Smoke is one fold with two training epochs, not a completed experiment.

The historical contract expects147 candidates/24 subjects; one candidate has a clipped EOF endpoint. See [protocol](../docs/PROTOCOL.md) before new scientific results. Window labels are inherited path labels. Technical QC, conditional accuracy and desktop latency do not establish instantaneous/clinical/Pi validity.
