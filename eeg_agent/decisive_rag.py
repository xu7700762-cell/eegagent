"""Optimized labelled RAG evidence; the final High/Low decision belongs to the LLM.

Pure retrieval functions extracted from the frozen experiment, with public tool
names. No EEG loading, private path dependency, query labels or fusion decisions.
Use validate_bank before context_from_bank when supplying your own evidence.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
from scipy.special import logit

FAMILIES = ("vrms_features", "spectral", "spatial_covariance", "vrms_mean",
            "vrms_temporal", "spectral_compact", "filterbank_csp", "case_retrieval")


def _check_roles(fold, split, subjects):
    roles = [set(split[key]) for key in ("base_train", "meta_calibration", "policy_validation")]
    if (split["outer_subject"] != fold or any(fold in role for role in roles)
            or any(roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3))
            or set(map(int, subjects)) != roles[2] or not roles[2]):
        raise ValueError("Policy reference subject isolation failed")


def _vector(probability, tools):
    if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not 0 <= probability <= 1:
        raise ValueError("Invalid MIL evidence")
    if len(tools) != len(FAMILIES) or {item["tool"] for item in tools} != set(FAMILIES):
        raise ValueError("Incomplete nine-output evidence")
    by_name = {item["tool"]: item for item in tools}
    result = np.asarray([logit(np.clip(probability, 1e-6, 1 - 1e-6))]
                        + [by_name[name]["raw_decision_score"] for name in FAMILIES], float)
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError("Nonfinite query/reference output")
    return result

FIELDS = ("MIL",) + FAMILIES
def _values(probability, tools):
    _vector(probability, tools)  # Validate complete actual finite values.
    by_name = {tool["tool"]: tool for tool in tools}
    raw = [probability] + [by_name[name]["raw_decision_score"] for name in FAMILIES]
    calibrated = [None]
    for name in FAMILIES:
        tool = by_name[name]
        usable, p = tool["calibration_usable"], tool["calibrated_high_probability"]
        if type(usable) is not bool or (usable and (isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1)) or (not usable and p is not None):
            raise ValueError("Invalid source calibration evidence")
        calibrated.append(p)
    direction = ["High" if raw[0] >= .5 else "Low"] + ["High" if score >= 0 else "Low" for score in raw[1:]]
    cal_direction = [None if p is None else "High" if p >= .5 else "Low" for p in calibrated]
    return raw, direction, calibrated, cal_direction


def validate_bank(bank, fold):
    """Check subject isolation, original readings and reference count tables.

    These checks verify numeric replay consistency. They do not regenerate EEG
    or independently attest the original model weights and data provenance.
    """
    if bank["outer_subject"] != fold or bank["schema"] != "raw_policy_current_raw_v3":
        raise ValueError("Wrong reference fold or schema")
    refs = bank["cases"]
    _check_roles(fold, bank["split"], [case["reference_subject"] for case in refs])
    if {case["reference_class"] for case in refs} != {"Low", "High"}:
        raise ValueError("Both reference classes required")
    vectors, raw, calibrated = [], [], []
    for case in refs:
        derived = _vector(case["mil_raw_high_probability"], case["individual_tools"])
        saved = np.asarray(case["retrieval_vector"], float)
        values = _values(case["mil_raw_high_probability"], case["individual_tools"])
        raw.append(values[0])
        calibrated.append(values[2])
        # The eight saved raw scores must match exactly. logit(MIL) is a
        # derived value and can differ by a last bit across SciPy runtimes.
        if (saved.shape != (9,) or not np.array_equal(derived[1:], saved[1:])
                or not np.isclose(derived[0], saved[0], rtol=1e-12, atol=1e-12)):
            raise ValueError("Reference vector differs from original readings")
        vectors.append(saved)
    vectors = np.asarray(vectors)
    expected_scale = np.maximum(np.percentile(vectors, 75, axis=0) - np.percentile(vectors, 25, axis=0), .1)
    if (not np.allclose(np.median(vectors, axis=0), bank["vector_median"], rtol=1e-12, atol=1e-12)
            or not np.allclose(expected_scale, bank["vector_scale"], rtol=1e-12, atol=1e-12)):
        raise ValueError("Reference normalization differs from bank readings")
    table = bank["no_rest_policy_confusion"]
    subjects = len({case["reference_subject"] for case in refs})
    labels = np.asarray([case["reference_class"] == "High" for case in refs])
    if (table["paths"] != len(refs) or table["subjects"] != subjects
            or table["true_Low_paths"] != int((~labels).sum())
            or table["true_High_paths"] != int(labels.sum())):
        raise ValueError("Wrong reference totals")
    expected_rows = {(name, "raw_probability_0.5" if name == "MIL" else "raw_score_zero") for name in FIELDS}
    for column, name in enumerate(FIELDS[1:], 1):
        available = [values[column] is not None for values in calibrated]
        if any(available) and not all(available):
            raise ValueError("Calibration availability differs within frozen fold")
        if all(available):
            expected_rows.add((name, "calibrated_probability_0.5"))
    if len(table["rows"]) != len(expected_rows) or {(row["tool"], row["boundary"]) for row in table["rows"]} != expected_rows:
        raise ValueError("Incomplete reference direction table")
    for row in table["rows"]:
        column = FIELDS.index(row["tool"])
        use_calibrated = row["boundary"] == "calibrated_probability_0.5"
        readings = [values[column] for values in (calibrated if use_calibrated else raw)]
        # Unavailable calibration remains null and has no calibrated count row.
        if any(value is None for value in readings):
            raise ValueError("Direction table requires available calibration")
        high = np.asarray(readings) >= (0 if row["boundary"] == "raw_score_zero" else .5)
        expected = [int((~labels & ~high).sum()), int((~labels & high).sum()),
                    int((labels & ~high).sum()), int((labels & high).sum())]
        actual = [row[key] for key in ("true_Low_output_Low", "true_Low_output_High", "true_High_output_Low", "true_High_output_High")]
        if expected != actual or row["paths"] != len(refs) or row["subjects"] != subjects:
            raise ValueError("Reference direction counts differ from readings")
    return bank

_CONTRASTIVE_SCHEMA = "raw_current_contrastive_rag_v2"
MAX_CASES = 8
SELECTION = "nearest_each_reference_class_by_eight_raw_tool_distance_then_nearest_each_reference_class_by_eight_raw_sign_pattern_then_cover_four_reference_subjects_then_distance_to_eight"

def _contrastive_context(query_prediction, bank, bank_sha, manifest_sha):
    numeric = query_prediction["numeric_fusion"]
    if numeric.get("reference_available") is not False:
        raise ValueError("Same no-rest branch required")
    qraw, qdir, qcal, qcaldir = _values(query_prediction["mil_raw_high_probability"], numeric["actual_tools"])
    query = np.asarray(qraw[1:], float)
    refs = bank["cases"]
    vectors = np.asarray([case["retrieval_vector"][1:] for case in refs])
    median, scale = np.asarray(bank["vector_median"][1:]), np.asarray(bank["vector_scale"][1:])
    if median.shape != (8,) or scale.shape != (8,) or not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid frozen eight-output normalization")
    delta = np.clip((vectors - median) / scale, -3, 3) - np.clip((query - median) / scale, -3, 3)
    distance = np.sqrt(np.mean(delta ** 2, axis=1))
    mismatch = np.sum((vectors >= 0) != (query >= 0), axis=1)
    ordered = np.argsort(distance, kind="stable").tolist()
    pattern_order = sorted(ordered, key=lambda i: (int(mismatch[i]), distance[i], i))
    chosen, reasons = [], {}

    def add(i, reason):
        if i is None:
            return
        reasons.setdefault(i, []).append(reason)
        if i not in chosen and len(chosen) < MAX_CASES:
            chosen.append(i)

    for label in ("Low", "High"):
        add(next((i for i in ordered if refs[i]["reference_class"] == label), None), "nearest_true_" + label + "_eight_tool_output")
    for label in ("Low", "High"):
        add(next((i for i in pattern_order if refs[i]["reference_class"] == label), None), "nearest_true_" + label + "_eight_raw_sign_pattern")
    all_subjects = sorted({case["reference_subject"] for case in refs})
    for i in ordered:
        present = {refs[offset]["reference_subject"] for offset in chosen}
        if refs[i]["reference_subject"] not in present:
            add(i, "different_internal_subject_coverage")
        if len({refs[offset]["reference_subject"] for offset in chosen}) == len(all_subjects):
            break
    for i in ordered:
        if i not in chosen:
            add(i, "remaining_nearest_eight_tool_output")
        if len(chosen) >= MAX_CASES:
            break
    cases, prototypes = [], []
    for rank, i in enumerate(chosen, 1):
        case = refs[i]
        raw, direction, calibrated, cal_direction = _values(case["mil_raw_high_probability"], case["individual_tools"])
        cases.append({"case_id": f"current-contrastive-case-{rank:02d}",
                      "subject_group": f"internal-subject-{all_subjects.index(case['reference_subject']) + 1:02d}",
                      "reference_class": case["reference_class"], "selection_reasons": reasons[i],
                      "raw_values": raw, "raw_directions": direction,
                      "calibrated_high_probabilities": calibrated, "calibrated_directions": cal_direction,
                      "raw_direction_matches_query": [a == b for a, b in zip(direction, qdir)],
                      "eight_tool_distance": float(distance[i]), "eight_tool_scaled_case_minus_query": delta[i].tolist()})
    for label in ("Low", "High"):
        members = [case for case in refs if case["reference_class"] == label]
        raw_matrix = np.asarray([_values(case["mil_raw_high_probability"], case["individual_tools"])[0] for case in members])
        prototypes.append({"citation_id": "current-policy-class-description-" + label,
                           "reference_class": label, "paths": len(members),
                           "subjects": len({case["reference_subject"] for case in members}),
                           "raw_medians": np.median(raw_matrix, axis=0).tolist(),
                           "raw_IQRs": (np.percentile(raw_matrix, 75, axis=0) - np.percentile(raw_matrix, 25, axis=0)).tolist()})
    table = bank["no_rest_policy_confusion"]
    rows = [[row["tool"], row["boundary"], row["true_Low_output_Low"], row["true_Low_output_High"], row["true_High_output_Low"], row["true_High_output_High"]]
            for row in table["rows"]]
    return {"schema": _CONTRASTIVE_SCHEMA, "available": True, "provenance": "fresh_same_fold_frozen_inference_on_saved_current_raw_policy_windows",
            "validation_scope": "head_and_calibrator_subject_heldout_selector_reused", "rest_reference_available": False,
            "retrieval_scope": "eight_raw_tool_outputs_only; MIL_excluded_from_distance_and_selection; reference_labels_internal_policy_only",
            "column_order": list(FIELDS), "query_reading": {"raw_values": qraw, "raw_directions": qdir,
                "calibrated_high_probabilities": qcal, "calibrated_directions": qcaldir},
            "field_definitions": {"raw_values": "Exact original MIL raw High sigmoid probability, followed by eight frozen original decision scores in column_order.",
                "raw_directions": "Each individual output only: MIL threshold 0.5; eight score thresholds zero; no combined query class.",
                "calibrated_high_probabilities": "Null means unavailable. MIL is null because its raw sigmoid output is not probability-calibrated.",
                "eight_tool_scaled_case_minus_query": "Eight coordinate differences after frozen policy-reference median/IQR standardization, IQR floor 0.1, coordinate clip [-3,3]. MIL is excluded.",
                "raw_direction_matches_query": "Per-coordinate comparison, not a combined vote or query confidence.",
                "class_descriptions": "True-Low/true-High medians/IQRs over all internal same-fold policy raw reference outputs; descriptive statistics, not a recommended query class.",
                "direction_error_counts": "All same-fold policy subjects, freshly encoded with current raw pipeline and no resting reference; counts, not query probabilities."},
            "retrieval": {"selection_rule": SELECTION, "candidate_paths": len(refs), "candidate_subjects": len(all_subjects),
                          "returned_paths": len(cases), "returned_subjects": len({case["subject_group"] for case in cases}),
                          "distance_metric": "RMS_eight_raw_tool_scores_reference_median_IQR_floor0.1_clip3; MIL_not_used"},
            "cases": cases, "class_descriptions": prototypes,
            "direction_error_counts": {"citation_id": "current-raw-policy-direction-errors", "paths": table["paths"], "subjects": table["subjects"],
                "true_Low_paths": table["true_Low_paths"], "true_High_paths": table["true_High_paths"],
                "columns": ["tool", "boundary", "true_Low_output_Low", "true_Low_output_High", "true_High_output_Low", "true_High_output_High"], "rows": rows},
            "review_guidance": {"citation_id": "current-raw-contrastive-review", "text":
                "Evaluate the true-High and true-Low reference hypotheses against the same-column query readings. References with similar eight-tool sign patterns can have opposite true labels: compare the nearest support and nearest counterexample, including each differing feature group. Use the full current-raw no-rest direction-error counts to check whether each individual raw/calibrated direction has produced both classes and how small its subject support is. Extreme raw MIL values and unanimous raw tool directions can still be wrong; no single output, tool majority, nearest label or prototype is authoritative. MIL is an attached actual reading and is excluded from retrieval. Shared VRMSModel/spectral/spatial sources and repeated paths within one reference subject are dependent. Historical policy reports containing resting references represent a different branch; current-raw tables here replay the same pipeline as the query. Reference selection is label-balanced for comparison, so returned-label frequency is not a query prior. Choose your final High/Low judgment independently from all actual evidence and state the strongest opposing evidence; do not copy an algorithmic vote, probability or recommendation."},
            "limitations": ["Internal policy subjects were excluded from MIL, seven EEG-head and calibrator fits but reused for selector development; no selector output is supplied.",
                            "The examples use the same current-raw preprocessing/event windows and outerfold frozen inference as the query; no fitting or source-query labels are used.",
                            "Only four reference subjects are available per fold; repeated paths, shared EEG features and task/domain drift limit transfer.",
                            "True class statistics and distances describe internal reference evidence, not current-query truth or calibrated confidence."],
            "source_hashes": {"current_raw_policy_bank_sha256": bank_sha, "current_raw_policy_manifest_sha256": manifest_sha,
                              "current_raw_worker_result_sha256": bank["source_hashes"]["current_raw_worker_result"]}}

SCHEMA = "raw_decisive_rag_v4"


VARIANTS = ("compact_same", "percentile_group")


GROUPS = {
    "VRMSModel": (0, 3, 4),
    "spectral": (1, 5),
    "spatial": (2, 6),
    "training_case_retrieval": (7,),
}


GUIDANCE = (
    "Compare the query with labelled Low and High demonstrations using the same columns. "
    "Explain which source groups and score magnitudes distinguish the closer pattern, and choose one class. "
    "Treat outputs within each source group jointly. Use raw and calibrated readings as different "
    "views of the same tool. Reference labels describe their own paths; your final class is your judgment. "
    "Base the decision on source-group pattern comparison, with no tool voting or computed query probability. "
    "Determine confidence from the strength of this comparison, separately from the final class."
)


SCOPE_NOTE = (
    "References use the query's frozen outerfold heads, current raw EEG window pipeline and no-rest branch. "
    "Their subjects were held out from heads and calibrators; these internal subjects also supported selector "
    "development. Rows sharing a subject or source group share evidence. Both reference classes are deliberately "
    "included for comparison, so the displayed class balance is a selection property."
)


def _compact(source, variant):
    """Keep original source readings; remove threshold-derived duplicate arrays."""
    cases = [{key: case[key] for key in
              ("case_id", "subject_group", "reference_class", "raw_values",
               "calibrated_high_probabilities", "eight_tool_distance")}
             for case in source["cases"]]
    return {
        "schema": SCHEMA, "variant": variant, "available": True,
        "provenance": source["provenance"], "validation_scope": source["validation_scope"],
        "retrieval_scope": source["retrieval_scope"], "rest_reference_available": False,
        "column_order": list(FIELDS),
        "query_reading": {key: source["query_reading"][key] for key in
                          ("raw_values", "calibrated_high_probabilities")},
        "reading_definitions": {
            "raw_values": "MIL raw High sigmoid followed by eight original decision scores. Thresholds: MIL 0.5; scores zero.",
            "calibrated_high_probabilities": "Individual tool calibration; null means this reading is unavailable.",
            "class_descriptions": "Exact whole-bank class medians and IQRs, in column_order.",
            "direction_error_counts": "Whole-bank direction counts, computed from the same no-rest inference.",
            "distances": "Reading-pattern similarity for selecting demonstrations.",
        },
        "source_groups": {group: [FAMILIES[i] for i in indices] for group, indices in GROUPS.items()},
        "retrieval": dict(source["retrieval"]), "cases": cases,
        "class_descriptions": source["class_descriptions"],
        "direction_error_counts": source["direction_error_counts"],
        "review_guidance": {"citation_id": "raw-decisive-comparison", "text": GUIDANCE},
        "scope_note": SCOPE_NOTE, "source_hashes": dict(source["source_hashes"]),
    }


def _percentiles(vectors, points):
    """Unlabelled empirical midranks; ties stay together, no score clipping."""
    return np.asarray([np.mean(vectors < point, axis=0) + .5 * np.mean(vectors == point, axis=0)
                       for point in points])


def _group_selection(refs, distance):
    ordered = sorted(range(len(refs)), key=lambda i: (float(distance[i]), i))
    subjects = {case["reference_subject"] for case in refs}
    candidates, options = {}, []
    for label in ("Low", "High"):
        candidates[label] = [i for subject in sorted(subjects) for i in
                             [j for j in ordered if refs[j]["reference_class"] == label
                              and refs[j]["reference_subject"] == subject][:2]]
        if not candidates[label]:
            raise ValueError("Both reference classes required")
        options.append(list(combinations(candidates[label], min(2, len(candidates[label])))))
    # Four core demonstrations: two per class, maximize distinct subjects,
    # then minimize similarity distance. The search has <=16 candidates.
    core = min((low + high for low, high in product(*options)), key=lambda indices: (
        -len({refs[i]["reference_subject"] for i in indices}), sum(float(distance[i]) for i in indices), indices))
    low = sorted((i for i in core if refs[i]["reference_class"] == "Low"), key=lambda i: distance[i])
    high = sorted((i for i in core if refs[i]["reference_class"] == "High"), key=lambda i: distance[i])
    chosen = [low[0], high[0]] + low[1:] + high[1:]
    # When a class exists in only one subject, add the nearest missing subject
    # instead of repeating five or more paths from an already represented one.
    for index in ordered:
        if len(chosen) >= 6 or len({refs[i]["reference_subject"] for i in chosen}) == len(subjects):
            break
        if refs[index]["reference_subject"] not in {refs[i]["reference_subject"] for i in chosen}:
            chosen.append(index)
    return chosen


def _pattern_rows(refs, vectors, query):
    rows = []
    subjects = sorted({case["reference_subject"] for case in refs})
    for group, indices in GROUPS.items():
        pattern = (query[list(indices)] >= 0).tolist()
        matches = np.all((vectors[:, list(indices)] >= 0) == pattern, axis=1)
        evidence = []
        for subject in subjects:
            members = [i for i, case in enumerate(refs) if case["reference_subject"] == subject and matches[i]]
            evidence.append([f"internal-subject-{subjects.index(subject) + 1:02d}",
                             sum(refs[i]["reference_class"] == "Low" for i in members),
                             sum(refs[i]["reference_class"] == "High" for i in members)])
        rows.append({"group": group, "query_raw_sign_pattern": ["High" if value else "Low" for value in pattern],
                     "same_pattern_reference_counts_by_subject": evidence})
    return {"citation_id": "raw-source-pattern-examples", "columns": ["subject_group", "reference_Low_paths", "reference_High_paths"], "rows": rows}


def context_from_bank(query_prediction, bank, provenance, variant="compact_same"):
    """Build context from an already source-verified bank (useful for batching)."""
    if variant not in VARIANTS:
        raise ValueError("Unknown decisive RAG variant")
    source = _contrastive_context(query_prediction, bank, provenance["current_raw_policy_bank_sha256"],
                              provenance["current_raw_policy_manifest_sha256"])
    result = _compact(source, variant)
    result["source_hashes"].update(provenance)
    if variant == "compact_same":
        return result
    refs = bank["cases"]
    raw = np.asarray([_values(case["mil_raw_high_probability"], case["individual_tools"])[0] for case in refs])
    vectors, query = raw[:, 1:], np.asarray(source["query_reading"]["raw_values"][1:])
    percentiles = _percentiles(vectors, np.vstack([vectors, query]))
    deltas = percentiles[:-1] - percentiles[-1]
    group_distance = np.asarray([np.sqrt(np.mean(deltas[:, list(indices)] ** 2, axis=1))
                                 for indices in GROUPS.values()]).T
    distance = np.sqrt(np.mean(group_distance ** 2, axis=1))
    chosen = _group_selection(refs, distance)
    subjects = sorted({case["reference_subject"] for case in refs})
    cases = []
    for rank, index in enumerate(chosen, 1):
        case = refs[index]
        values, _, calibrated, _ = _values(case["mil_raw_high_probability"], case["individual_tools"])
        cases.append({"case_id": f"group-percentile-case-{rank:02d}",
                      "subject_group": f"internal-subject-{subjects.index(case['reference_subject']) + 1:02d}",
                      "reference_class": case["reference_class"], "raw_values": values,
                      "calibrated_high_probabilities": calibrated,
                      "group_percentile_distance": float(distance[index]),
                      "source_group_distances": dict(zip(GROUPS, group_distance[index].tolist()))})
    result["cases"] = cases
    result["query_reading"]["eight_raw_tool_reference_percentiles"] = percentiles[-1].tolist()
    result["reading_definitions"]["reference_percentiles"] = "Unlabelled empirical midrank among all same-fold reference scores; one entry per raw tool."
    result["reading_definitions"]["group_percentile_distance"] = "RMS over four source-group RMS differences in eight unlabelled empirical percentiles; MIL excluded."
    result["retrieval"] = {"selection_rule": "two_each_reference_class_max_distinct_subjects_then_min_distance_then_only_missing_subject_coverage_max6",
                           "candidate_paths": len(refs), "candidate_subjects": len(subjects),
                           "returned_paths": len(cases), "returned_subjects": len({case["subject_group"] for case in cases}),
                           "distance_metric": "equal_source_group_RMS_unlabelled_raw_tool_midrank; MIL_excluded"}
    result["group_sign_pattern_reference_counts"] = _pattern_rows(refs, vectors, query)
    result["retrieval_scope"] = "eight_raw_tools_unlabelled_percentiles_equal_source_groups; MIL_excluded; reference_labels_internal_policy_only"
    return result
