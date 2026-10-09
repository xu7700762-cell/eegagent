# Numerical pilot V2

`data.py` audits real CEO/Trigger task boundaries before cache allocation, preserving excluded candidates. `model.py` trains the compact CNN from scratch. `reliability.py` fits independent Platt calibration and empirical policy gates. `engine.py` and `experiment.py` perform four-state decisions and lazy evidence acquisition.

The independent V3 experiment uses `reliability_v3.py`, `engine_v3.py`, and `reliability_experiment_v3.py` while preserving the frozen V2 sources. It cross-fits five meta-subject heads, separates policy selection from auditing and technical QC from coverage warnings. `model_ablation_v3.py` compares frozen CNN/FEMBA with path-level attention MIL. See [V3 protocol and actual results](../docs/RELIABILITY_V3.md). No cloud API or RAG expansion is involved, and no unvalidated V3 gate replaces the Cloud default.

Every outer LOSO fold keeps14 base /5 meta /4 policy /1 test subjects. Meta is split into3 head-fit and2 probability-calibration subjects. OOD reference descriptors fit base signals only; uncertainty/OOD/QC-fraction cutoffs are selected only on independent policy subjects. Missing/failed validation abstains. Adaptive probability stays p_cal regardless of cases, auxiliary probabilities or LLM explanations.

```bash
python -m vrms_pilot.experiment --stage audit --data-root /path/to/data
python -m vrms_pilot.experiment --stage prepare --data-root /path/to/data
python -m vrms_pilot.experiment --stage smoke
python -m vrms_pilot.experiment --stage run
python -m vrms_pilot.validate_results --data-root /path/to/data
```

Default output: `outputs/vrms_pilot/v2_seed2026`. Audit is read-only and writes eligibility records without EEG caches/training. Smoke is one fold/two epochs and remains separate from full results. Existing caches/results require a fresh directory. Historical147-path caches are rejected; current eligibility audit has146 complete paths/24 subjects.

ToolReplay computes QC/deep/reliability first; difficult cases request CaseRetriever, then physiology if retrieval is unvalidated/unreliable/conflicting. Raw deep/temporal+QC retrieval avoids running expensive physiology before it is requested. Spectral changes/covariance distance are unsigned physiological descriptions; verification remains inconclusive without directional validation.

Window labels are weak path labels. Conditional accuracy, engineering QC and desktop latency do not establish clinical/instantaneous symptoms or Pi validity. See [protocol](../docs/PROTOCOL.md).
