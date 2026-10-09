# Reliability V3 and frozen-encoder model comparison

V3 is an independent numerical experiment. Its objective is higher publication coverage only after independently checked prediction risk. It calls no cloud model and does not change RAG. The existing V2 source and artifacts remain intact so their frozen checks continue to pass.

## Code and roles

- `vrms_pilot/reliability_v3.py`: meta subject-OOF calibration, technical QC versus coverage warnings, fixed policy selection/audit, nested subject audits and multiple rejection reasons.
- `vrms_pilot/engine_v3.py`: four-state numeric decisions; explanations and auxiliary tools cannot change probabilities.
- `vrms_pilot/reliability_experiment_v3.py`: reloads the existing CNNs and original 24 LOSO splits, cross-fits scalar heads and saves all outer predictions and provenance.
- `vrms_pilot/risk_coverage_v3.py` and `analysis_v3.py`: held-out descriptive risk curves, per-fold AUROC/Brier/ECE and paired subject bootstrap diagnostics.
- `vrms_pilot/model_ablation_v3.py`: native frozen FEMBA and frozen CNN features with the same attention-MIL structure.

Each outer fold retains 14 base, 5 meta, 4 policy and 1 test subject. CNN weights and input normalization are unchanged. For every meta subject, a scalar head fits the other four and predicts that held-out subject. Platt consumes the five combined OOF predictions. The production scalar head fits all five meta subjects. The head's training size changes from V2, so this is not a pure calibrator-only comparison. Calibrator re-predictions on its own OOF fitting rows do not independently demonstrate calibration quality.

Policy IDs are sorted before inspecting outcomes: the first two select one candidate and the last two can only audit or reject it. Failed audits never cause reselection. Four additional 3-select/1-audit subject folds provide a stability veto. Candidates must meet the existing accuracy, coverage and sample goals as well as true/predicted class support, independent subject support, subject-macro error and accepted-set mean implied risk. These are exploratory operating objectives, not finite-sample or clinical risk bounds.

Base-only, label-independent valid-window proportions establish a coverage reference. Falling below that range can abstain with a warning; it alone cannot turn usable EEG into `insufficient_data`. Missing pre-task rest is reported but is not a technical QC failure.

## Run and verify

Use a scientific Python environment with the existing completed V2 artifacts. Defaults refer to the actual 2026-10-08 run. The V3 runner verifies the source against the earlier frozen Cloud preparation anchor before recording its own hashes. Reading that anchor makes no API call.

```bash
python -m unittest vrms_pilot.test_reliability_v3 vrms_pilot.test_reliability_experiment_v3 vrms_pilot.test_model_ablation_v3 -v
python -m vrms_pilot.reliability_experiment_v3 --stage validate
# A new experiment must use a fresh output directory.
python -m vrms_pilot.reliability_experiment_v3 --stage run --out outputs/vrms_reliability/v3_reproduce_seed2026
```

For another V2 run, pass `--source` and a matching `--anchor` Cloud preparation protocol. Neither the original source hashes nor the source results may be edited to make validation pass. V3 checkpoints, features and OOF predictions have their own artifact manifest; validation reloads the gates, regenerates meta OOF head predictions and recomputes outer features from the frozen source CNN.

The FEMBA stages require CUDA and the matching native `mamba-ssm` encoder runtime. Supply the encoder source and the declared matching checkpoint explicitly; neither is bundled in this release:

```bash
python -m vrms_pilot.model_ablation_v3 --stage preflight --encoder-source /path/to/encoder.py --pretrained-checkpoint /path/to/femba.ckpt --out outputs/vrms_model_ablation/v3_reproduce_seed2026
python -m vrms_pilot.model_ablation_v3 --stage extract --out outputs/vrms_model_ablation/v3_reproduce_seed2026
python -m vrms_pilot.model_ablation_v3 --stage run --out outputs/vrms_model_ablation/v3_reproduce_seed2026
python -m vrms_pilot.model_ablation_v3 --stage validate --out outputs/vrms_model_ablation/v3_reproduce_seed2026
```

## Actual 2026-10-08 result

The numerical V3 completed all 24 folds and 146 paths. Seven folds fitted calibration; no primary publication candidate passed selection. All 146 paths remain uncertain and the released subset is empty. Empty-subset risk is undefined, not zero. The old path 57 now stays uncertain rather than being called insufficient solely for its 23/24 valid-window proportion. This experiment corrected the protocol but did not demonstrate improved reliable coverage.

The inherited `failed_checks` list records fold-level candidate/audit conditions, including unavailable risk. It is not a set of measured per-path outer errors. The report separates actual path rejections from inherited policy diagnostics and records zero selected primary candidates and zero accepted audit samples.

For the same 38 scoreable V3 paths, raw/cal AUROC is 0.5722/0.5806, Brier 0.2704/0.2730 and ECE 0.2657/0.2068. Monotone calibration preserves each dual-class fold's AUROC. Missing or single-class fold AUROC stays null; Brier/ECE remain reportable on single-class data.

Frozen CNN+MIL and frozen FEMBA+MIL both scored all 146 paths, with BACC 50.11% and 61.15%, AUROC 0.5448 and 0.6549. The paired subject-bootstrap BACC difference interval is -3.78 to +25.37 percentage points. Both MILs are uncalibrated and overconfident. FEMBA's error at 30/146 descriptive coverage is 20%, with a wide subject interval; that score-ranked diagnostic is not a deployed gate.

FEMBA uses a local recovered auxiliary-pretraining encoder, not official TUH weights. All 83 tensors load strictly, but the original pretraining manifest and robust input statistics are missing. Pretraining overlap is not excluded. Encoder structures and MIL input projection parameter counts differ, so the experiment does not isolate pretraining as the only cause of a difference.

All new outputs reside in `outputs/vrms_reliability/v3_20261008` and `outputs/vrms_model_ablation/v3_20261008`. The Chinese report and standalone PNG/PDF/SVG curves retain denominators, missing values, source tables and exploratory confidence intervals. V3 is not switched into the Cloud default: no independently validated replacement gate has been established.
