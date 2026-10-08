"""Subject-diverse case retrieval; class counts are evidence, not probabilities.

Only raw deep/temporal summaries and QC enter the distance. Case RAG therefore
does not need to run spectral or covariance tools before it is requested.
"""
import copy

import numpy as np


TEMPORAL_FIELDS = ("raw_mean_probability", "raw_probability_std", "raw_recent_mean",
                   "raw_recent_slope_per_second", "raw_last_probability")
FEATURE_NAMES = (*TEMPORAL_FIELDS, "qc_accepted_fraction")


def case_vector(evidence):
    temporal = evidence.get("temporal") or {}
    values = [temporal.get(name) for name in TEMPORAL_FIELDS]
    values.append(evidence.get("qc_accepted_fraction"))
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def _scale(records):
    if not records:
        return np.ones(len(FEATURE_NAMES))
    x = np.stack([case_vector(record["evidence"]) for record in records])
    return np.asarray([max(float(np.std(column[np.isfinite(column)])), 1e-4)
                       if np.isfinite(column).any() else 1. for column in x.T])


def _neighbors(query, records, k, mask):
    """Fit feature scale on this bank only, before looking at neighbor labels."""
    q, scale = case_vector(query), _scale(records)
    candidates = []
    for record in records:
        row = case_vector(record["evidence"])
        common = mask & np.isfinite(q) & np.isfinite(row)
        if not common[:len(TEMPORAL_FIELDS)].any():
            continue
        distance = float(np.mean(((row[common] - q[common]) / scale[common]) ** 2))
        candidates.append((distance, record["path_index"], int(common.sum()), record))
    candidates.sort(key=lambda value: (value[0], value[1]))
    chosen, seen = [], set()
    for candidate in candidates:
        subject = candidate[-1]["subject"]
        if subject not in seen:
            chosen.append(candidate)
            seen.add(subject)
        if len(chosen) == k:
            break
    return chosen


def _vote(neighbors):
    high = sum(item[-1]["label"] for item in neighbors)
    low = len(neighbors) - high
    return high, low, (1 if high > low else 0 if low > high else None)


class CaseRetriever:
    """Frozen internal bank with lazy, query-mask-matched LOSO range selection.

    The range threshold is selected from meta-subject-held-out predictions;
    it is not a fixed EEG distance and is not a clinical reliability guarantee.
    No threshold is produced when the predeclared accuracy/coverage gates fail.
    For a query from the bank, its entire subject is excluded from neighbors,
    scaling and range selection, including all of that subject's labels.
    """
    def __init__(self, bank, evaluation, k=5, allowed_subjects=None,
                 target_accuracy=.80, minimum_cases=8, minimum_coverage=.35):
        if k < 1 or minimum_cases < 1 or not 0 < target_accuracy <= 1 or not 0 < minimum_coverage <= 1:
            raise ValueError("Invalid predeclared retrieval policy")
        self.k = k
        self.target_accuracy = target_accuracy
        self.minimum_cases = minimum_cases
        self.minimum_coverage = minimum_coverage
        self.records, self._calibrations = [], {}
        seen = set()
        for record in bank:
            if record.get("role") != "meta":
                raise ValueError("Case RAG accepts only internal meta records")
            index = int(record["path_index"])
            if index in seen:
                raise ValueError("Duplicate reference case")
            seen.add(index)
            source = evaluation[index]
            subject, label = source["subject_key"], source["label"]
            if isinstance(subject, np.generic):
                subject = subject.item()
            if allowed_subjects is not None and subject not in allowed_subjects:
                raise ValueError("Reference subject is outside the internal bank")
            if label not in (0, 1):
                raise ValueError("Reference class must be binary")
            self.records.append(dict(path_index=index, subject=subject, label=int(label),
                                     evidence=copy.deepcopy(record["evidence"])))

    def _calibration(self, records, mask):
        key = (tuple(record["path_index"] for record in records), tuple(mask))
        if key in self._calibrations:
            return self._calibrations[key]
        rows, folds = [], []
        subjects = sorted({record["subject"] for record in records})
        for subject in subjects:
            training = [record for record in records if record["subject"] != subject]
            target = [record for record in records if record["subject"] == subject]
            folds.append(dict(heldout_subject=subject,
                              training_subjects=sorted({record["subject"] for record in training}),
                              training_indices=[record["path_index"] for record in training],
                              target_indices=[record["path_index"] for record in target]))
            for record in target:
                neighbors = _neighbors(record["evidence"], training, self.k, mask)
                high, low, vote = _vote(neighbors)
                rows.append(dict(path_index=record["path_index"], nearest_distance=neighbors[0][0] if neighbors else None,
                                 high_count=high, low_count=low, vote_label=vote,
                                 correct=vote is not None and vote == record["label"]))
        threshold, selected = None, None
        candidates = sorted({row["nearest_distance"] for row in rows if row["nearest_distance"] is not None})
        for distance in candidates:
            accepted = [row for row in rows if row["nearest_distance"] is not None
                        and row["nearest_distance"] <= distance]
            accuracy = sum(row["correct"] for row in accepted) / len(accepted)
            coverage = len(accepted) / len(rows)
            if (len(accepted) >= self.minimum_cases and coverage >= self.minimum_coverage
                    and accuracy >= self.target_accuracy):
                threshold = float(distance)
                selected = dict(cases=len(accepted), coverage=coverage, neighbor_vote_accuracy=accuracy)
        result = dict(distance_threshold=threshold, status="validated_internal_range" if threshold is not None else "unvalidated",
                      source="meta-only subject-LOSO; scaler refitted excluding each heldout subject",
                      policy=dict(k=self.k, target_accuracy=self.target_accuracy, minimum_cases=self.minimum_cases,
                                  minimum_coverage=self.minimum_coverage), selected=selected,
                      selection_metrics_are_not_independent_final_validation=True,
                      folds=folds, rows=rows)
        self._calibrations[key] = result
        return result

    def __call__(self, query_evidence, query_subject=None, p_cal=None):
        records = [record for record in self.records if record["subject"] != query_subject]
        mask = np.isfinite(case_vector(query_evidence))
        calibration = self._calibration(records, mask)
        neighbors = _neighbors(query_evidence, records, self.k, mask)
        high, low, vote = _vote(neighbors)
        nearest = neighbors[0][0] if neighbors else None
        threshold = calibration["distance_threshold"]
        expected = min(self.k, len({record["subject"] for record in records}))
        complete = bool(neighbors) and len(neighbors) == expected
        reliable = bool(complete and threshold is not None and nearest <= threshold)
        status = ("insufficient_data" if not neighbors else "unvalidated" if threshold is None else
                  "reliable" if reliable else "unreliable")
        if p_cal is None:
            p_cal = query_evidence.get("p_cal", (query_evidence.get("reliability") or {}).get("p_cal"))
        conflict = (None if vote is None or p_cal is None or not np.isfinite(p_cal)
                    else bool(vote != int(p_cal >= .5)))
        quality = dict(status=status, reliable=reliable, mode="top_k_subject_diverse", k=self.k,
                       retrieved_count=len(neighbors), high_count=high, low_count=low,
                       distinct_subject_count=len(neighbors), nearest_distance=nearest,
                       distances=[item[0] for item in neighbors],
                       shared_feature_counts=[item[2] for item in neighbors],
                       available_query_features=[name for name, keep in zip(FEATURE_NAMES, mask) if keep],
                       distance_metric="mean squared standardized shared deep-temporal/QC features",
                       distance_threshold=threshold, outside_reliable_range=None if threshold is None or nearest is None else nearest > threshold,
                       calibration_source=calibration["source"], mixed_evidence=bool(high and low),
                       class_counts_are_not_calibrated_probabilities=True,
                       conflict_with_deep_model=conflict)
        examples = [dict(evidence=copy.deepcopy(item[-1]["evidence"]),
                         observed_class="high" if item[-1]["label"] else "low",
                         retrieval_distance=item[0], shared_feature_count=item[2]) for item in neighbors]
        return dict(examples=examples, selected_indices=[item[1] for item in neighbors],
                    retrieval_quality=quality, evidence_conflict=bool((high and low) or conflict),
                    conflict_with_deep_model=conflict)

    def calibration_audit(self):
        return self._calibration(self.records, np.ones(len(FEATURE_NAMES), dtype=bool))

    def loso_evaluation(self):
        """Nested subject-LOSO evaluation of the selected range and raw neighbors.

        Each evaluation subject is excluded from range selection as well as
        retrieval. These results can compare retrieval controls; no API is used.
        """
        rows = []
        for record in self.records:
            result = self(record["evidence"], query_subject=record["subject"])
            quality = result["retrieval_quality"]
            vote = 1 if quality["high_count"] > quality["low_count"] else 0 if quality["low_count"] > quality["high_count"] else None
            rows.append(dict(path_index=record["path_index"], heldout_subject=record["subject"],
                             selected_indices=result["selected_indices"], retrieval_quality=quality,
                             observed_class="high" if record["label"] else "low", vote_label=vote,
                             correct=vote is not None and vote == record["label"]))
        accepted = [row for row in rows if row["retrieval_quality"]["reliable"]]
        return dict(source="nested meta-only subject-LOSO; query subject absent from threshold selection",
                    cases=len(rows), neighbor_vote_accuracy=sum(row["correct"] for row in rows) / len(rows) if rows else None,
                    reliable_cases=len(accepted), coverage=len(accepted) / len(rows) if rows else 0.,
                    reliable_neighbor_vote_accuracy=sum(row["correct"] for row in accepted) / len(accepted) if accepted else None,
                    rows=rows, counts_are_not_probabilities=True)
