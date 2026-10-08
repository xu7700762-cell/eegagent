# Protocol and known limits

## Data contract

The existing pilot is a dataset-specific, single-seed (2026) experiment. Its historical cache contract expects 24 subjects and 147 candidate paths, with path scores `<30` / `>=30` (71/76). It is not a generic arbitrary-dataset loader.

```text
data-root/
  raw/Acquisition xx.cdt
  raw/Acquisition xx.cdt.dpo
  raw/Acquisition xx.cdt.ceo
  labels/task_segments_with_path_scores.csv
  labels/state_segments_by_mark.csv
```

CDT is little-endian float32, sample-major, 1024 Hz; DPO supplies 37-channel metadata and microvolt units. The 30 EEG channels exclude M1/M2, which provide the simultaneous reference. P0 uses a causal 50 Hz notch, 0.5–45 Hz bandpass, antialias filter and 1024→256 Hz decimation with preserved state. Model windows have shape 30×1280 and last 5 seconds. QC is an engineering sanity gate, not a clinically validated artifact detector.

Each scored path needs a score, at least 15 seconds and complete task boundaries. Initial reference features require a complete rest before the first task, at least 10 seconds after warm-up and at most 30 seconds. Missing reference features remain unavailable.

The label CSV needs `file`, `subject_id`, `is_complete`, `path_score_available`, `duration_sec`, `start_sample_1024`, `end_sample_1024`, `path_score`. The state CSV needs `subject_id`, `state`, `is_complete`, `start_sample_1024`, `end_sample_1024`. Use real audited event-to-score alignment; do not manufacture these labels from the synthetic example.

## Known event-completeness problem

The historical loader trusts CSV completeness flags and checks only that its saved endpoint fits the raw file. An audited candidate's endpoint was clipped to EOF while its actual ending event occurred 18.8984 seconds later. The historical 147-path evaluation therefore includes a truncated candidate; 146 paths have complete raw events. A separate recording correctly uses Trigger-channel fallback rather than CEO events.

This source release documents, but does not silently repair, the historical experiment. Excluding the truncated path requires new eligibility/cache artifacts and rerunning affected train/meta/policy stages. Removing one final OOF row does not repair its participation in other folds. The `147/24` and downstream request-count guards still reflect the old frozen contract and must be deliberately revised together for a corrected experiment.

## Splits and numerical candidates

Outer evaluation leaves one subject out in each of 24 folds. The other subjects are split into 14 base-training, 5 meta/calibration and 4 policy-validation subjects. Base models exclude meta/policy/test subjects; meta heads exclude policy/test subjects. The policy set selects the final numerical candidate by BACC. Whole-path labels supervise windows weakly; window scores are not momentary symptom truth.

Candidates are original full evidence, regional spectrum, spatial spectrum, training-reference tangent covariance, path MIL CNN and their mean. MIL samples 16 windows per path, averages logits and applies one path-level BCE term. Epochs 8/16/32 are selected using meta labels; window normalization, dropout .35 and AdamW regularization are jointly changed. The measured gain is not an isolated MIL ablation.

Refinement cross-fits calibration heads, not the whole frozen encoder/epoch selection. Its policy CV is conditional on a numerical candidate already selected with all four policy subjects. Outer data have been examined repeatedly and remain exploratory.

## Agent contract

Independent requests contain one anonymous query and internal examples only. Evidence includes numerical probabilities, temporal summaries, four-band power summaries, initial-reference changes, unsigned covariance distance, QC fraction and reference availability. Ground truth and source identifiers are rejected in query evidence. Example labels come only from internal meta subjects.

The model returns `id`, `high_probability`, `state`, `uncertain`, `reason`. State must agree with probability threshold .5. The fixed comparison mixes numerical/cloud scores .75/.25; failures retain the numeric output and are logged. The cloud cannot choose its own blend weight. Uncertainty flags and `max(p, 1-p)` are not calibrated confidence.

GPT per-fold batches are preserved as a separate diagnostic context; they are not online causal single-path requests. The weight sweep selects maxima using already examined outer labels and is explicitly post hoc. Its intervals do not correct this search bias.

## Reporting and release scope

Report path ACC, BACC, AUROC, Brier and coverage separately; covariance's available-reference subset and published-only accuracy cannot be ranked directly against all-path metrics. Bootstrap resamples subjects, not windows. Logical API success counts include recovery attempts; actual attempts and failure history must remain visible.

Historical point estimates were 63.95% for the selected numerical procedure, 65.99% for original GPT fusion and 63.27% for original DeepSeek fusion. GPT refinement's primary result was 63.95%, DeepSeek refinement 66.67%; both accuracy/calibration promotion gates failed. The secondary GPT calibrated result 67.35% did not pass its calibration gate.

This release contains no original recordings, scores, checkpoints, per-subject predictions or API responses. It includes no Pi hardware measurements, complete streaming service or official BrainAgent implementation. Public contract tests verify engineering properties; they do not establish clinical/generalization validity or repair the eligibility problem.
