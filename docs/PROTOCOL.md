# V2 protocol and evidence boundaries

## Data eligibility

The endpoint is a whole-path score `<30` / `>=30`. Window labels are weak path labels, not instantaneous symptoms. CDT is float32 little-endian, sample-major, 1024 Hz, 37 channels; the model uses 30 EEG channels excluding M1/M2, which provide the simultaneous average reference. Stateful causal notch/bandpass/antialias processing yields 256 Hz, 5-second windows.

The data root must contain `raw/Acquisition xx.cdt[.dpo/.ceo]` and `labels/task_segments_with_path_scores.csv`, `labels/state_segments_by_mark.csv`. Task rows need file, subject_id, completeness/score flags, duration, start/end raw samples and path_score. Event completeness is verified against the actual task marks, not only CSV flags or a clipped endpoint.

`audit_eligibility` retains all scored candidate records and exclusion reasons. CEO task marks are primary; if absent, Trigger channel transitions are used. The first event after a start20 must be the corresponding end22, exactly matching the CSV endpoint and within actual raw samples. The known acquisition04 candidate starts2516402, ends at clipped CSV EOF2626240, while its actual end22 is2645592. Current read-only audit gives147 candidates,146 eligible paths,24 subjects. Preparation allocates caches only after this filtering. Counts in downstream V2 stages are manifest-derived.

An initial complete rest before the first task supplies a reference only if at least10 seconds remain after warm-up; at most30 seconds are used. Missing references stay unavailable. QC amplitude/nonflatness checks are engineering sanity constraints; the additional fraction acceptance gate is learned on internal policy subjects. QC has not been proven to detect every artifact.

## Subject roles and calibration

Each of24 outer LOSO folds has14 base-training,5 meta,4 policy-validation and1 outer-test subject. The5 meta subjects are partitioned into3 head-fit and2 probability-calibration subjects with class support checked internally. CNN and physiological classifiers fit base subjects only, with fixed epochs. Classification heads fit the3 head-fit subjects. A monotone Platt calibrator fits deep-head predictions from the other2 meta subjects. There is no calibration on the same head-fit prediction rows.

OOD features are cheap log channel-scale/amplitude descriptors; robust center and scale use base-training signals only. Margin, OOD distance and valid-window-fraction cutoffs are empirical internal policy-set values. The declared selection goals are accuracy>=.80, coverage>=.35 and at least8 accepted paths. These are exploratory goals, not a reliability guarantee. Missing classes, failed calibration, reversed calibration slope or unmet policy goals prevent a reliable class release.

The four outputs are high/low/uncertain/insufficient_data. Final probability is p_cal exclusively; p_raw is separate. A usable but uncertain/OOD input remains uncertain; poor signal quality can yield insufficient_data. No valid calibration produces p_cal=null. Auxiliary tool classifiers and their consensus are diagnostic, not probability updates.

## Dynamic evidence and Case RAG

ToolReplay memoizes actual operations and records calls/timing. The deep+quality reliability gate can exit before retrieval/physiology. Otherwise CaseRetriever is called; unreliable/unvalidated retrieval or mixed/conflicting cases trigger Biomarker and available-reference Covariance. These tools do not overwrite p_cal. V2 cloud.prepare only builds deep/temporal/QC snapshots. Offline preparation timing is not the runtime lazy latency.

Retrieval uses shared finite raw deep/temporal/QC coordinates standardized on the internal case bank. Rankings are not label constrained. K=5, one case per subject, from the5 meta subjects; query subjects are excluded. Returned distances, high/low counts, distinct subject count, feature availability and range status are descriptions. Class count fractions are not calibrated outcome probabilities.

Reliable distance ranges are selected from meta-only subject-LOSO neighbor votes with scaler refits excluding each heldout subject. Range evaluation uses a nested LOSO that also excludes the evaluation subject from threshold selection. If no internal threshold passes the declared goals, the retrieval is unvalidated. Per-fold internal banks overlap across outer folds, so these diagnostics cannot be counted as independent outer evidence. Balanced up-to4-per-class few-shot selection is retained as a separate control.

## Physiological and cloud evidence

Biomarker descriptions report mean relative delta/theta/alpha/beta power and natural-log changes from the initial pre-task reference. Covariance distance is the mean Frobenius norm of the log reference-whitened covariance. Its magnitude has no high/low direction. No fixed spectral direction rule is used. Without independent directional verification the structured verification stays inconclusive with null model conflict. Missing reference is not low-class evidence.

The cloud schema returns only id, supporting_evidence, conflicting_evidence, missing_evidence and explanation. It cannot return probabilities, final states or confidence. Supervisor determines the final state from machine reliability. Explanations, failures or injected scores cannot change p_cal or release an unvalidated class. Reliable and quality-rejected cases skip cloud calls. Requests contain one anonymous query, internal examples and no query labels/source identities/raw EEG. Local audits may keep subject provenance; outbound reliability contains counts only.

## Reporting and preserved history

Report all eligible-path denominators, score coverage, publication coverage, conditional published metrics, Brier/AUROC and model-threshold diagnostics separately. Refusals remain in the denominator. Subject-cluster bootstrap is appropriate for comparisons; windows are not independent replicates. Tool timing is desktop assessment after preprocessing, excluding recording time, complete deployment latency, clinical/Pi validity and any skipped computation.

V2 defaults use independent v2_seed2026 directories. Existing caches/training results are not overwritten. Old147-path caches/checkpoints are rejected and must be regenerated after eligibility correction. Merely dropping the old final OOF row does not repair earlier model fitting or selection. V1 code/results and release manifests remain historical references (commit205abc7), with no V2 performance claim. No formal retraining or real API experiment was run for this change. Previously inspected outer data remain exploratory.