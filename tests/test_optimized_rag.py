"""Protect actual context compatibility, label isolation and response preservation."""
import copy
import json
import shutil

import pytest

from eeg_agent.decisive_rag import _percentiles, context_from_bank, validate_bank
from eeg_agent.rag_experiment import RESULTS, load_replay, parse_completed_response, payload_for, validate_final
from scripts.verify_optimized_rag import verify_results


@pytest.fixture(scope="module")
def replay():
    return load_replay()


def test_all146_frozen_contexts_and_actual_paired_scores():
    result = verify_results()
    assert result["contexts_verified"] == 146
    assert result["metrics"]["without_rag"]["correct"] == 104
    assert result["metrics"]["percentile_group"]["correct"] == 107
    assert result["paired"] == {"both_valid": 146, "corrected": 12, "damaged": 9, "net": 3}
    assert result["target_plus_four_met"] is False
    assert result["confidence"]["without_rag"]["high"] == {"total": 38, "correct": 32, "accuracy": 32 / 38}
    assert result["confidence"]["percentile_group"]["high"] == {"total": 40, "correct": 34, "accuracy": .85}


def test_query_truth_and_fusion_are_ignored_for_every_query(replay):
    inputs, banks, _, _ = replay
    for query in inputs.values():
        entry = banks[str(query["fold"])]
        original = context_from_bank(query["prediction"], entry["bank"], entry["historical_provenance"], "percentile_group")
        mutated = copy.deepcopy(query["prediction"])
        mutated.update(true_class="opposite", questionnaire_score=999, previous_answer="opposite")
        mutated["numeric_fusion"].update(predicted_class="opposite", raw_combined_score=999)
        assert original == context_from_bank(mutated, entry["bank"], entry["historical_provenance"], "percentile_group")


def test_mil_changes_reading_but_never_selected_cases_or_distance(replay):
    inputs, banks, _, _ = replay
    for query in inputs.values():
        entry = banks[str(query["fold"])]
        original = context_from_bank(query["prediction"], entry["bank"], entry["historical_provenance"], "percentile_group")
        mutated = copy.deepcopy(query["prediction"])
        mutated["mil_raw_high_probability"] = 1 - mutated["mil_raw_high_probability"]
        changed = context_from_bank(mutated, entry["bank"], entry["historical_provenance"], "percentile_group")
        assert original["cases"] == changed["cases"]
        assert original["group_sign_pattern_reference_counts"] == changed["group_sign_pattern_reference_counts"]


def test_reference_subject_leakage_and_changed_vectors_rejected(replay):
    _, banks, _, _ = replay
    for entry in banks.values():
        bank = copy.deepcopy(entry["bank"])
        bank["cases"][0]["reference_subject"] = bank["outer_subject"]
        with pytest.raises(ValueError, match="isolation"):
            validate_bank(bank, bank["outer_subject"])
    bank = copy.deepcopy(banks["1"]["bank"])
    bank["cases"][0]["retrieval_vector"][1] += .1
    with pytest.raises(ValueError, match="vector"):
        validate_bank(bank, 1)


def test_unavailable_calibration_has_no_calibrated_direction_row(replay):
    _, banks, _, _ = replay
    # Some folds genuinely lack calibration; null is preserved, not made zero.
    bank = banks["5"]["bank"]
    assert len(bank["no_rest_policy_confusion"]["rows"]) < 17
    assert validate_bank(bank, 5) is bank


def test_derived_mil_last_bit_difference_is_tolerated_but_raw_changes_are_not(replay):
    _, banks, _, _ = replay
    bank = copy.deepcopy(banks["1"]["bank"])
    bank["cases"][0]["retrieval_vector"][0] += 1e-14
    assert validate_bank(bank, 1) is bank
    bank["cases"][0]["retrieval_vector"][1] += 1e-14
    with pytest.raises(ValueError, match="vector"):
        validate_bank(bank, 1)


def test_modified_public_evidence_rejected_before_scoring(tmp_path):
    for path in RESULTS.glob("optimized_rag_*"):
        shutil.copyfile(path, tmp_path / path.name)
    path = tmp_path / "optimized_rag_paths.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Replay file changed"):
        load_replay(tmp_path)


def native_final():
    return {"final_class": "High", "confidence": "medium", "explanation": "依据工具证据判断。",
            "supporting_tools": ["initial_evidence", "baseline"], "counterevidence_tools": ["vrms_mean"],
            "rag_citations": ["unknown-case"]}


def test_citation_warning_preserves_all_native_fields_without_resampling(replay):
    inputs, banks, _, manifest = replay
    payload = payload_for("q001", "without_rag", inputs, banks, manifest)
    final = native_final()
    preserved, warnings = validate_final(final, payload)
    assert preserved == final and preserved is not final
    assert {warning["kind"] for warning in warnings} == {"unknown_tool_reference", "known_MIL_input_alias", "unknown_RAG_reference"}
    # Validating the alternate class uses the same rule and cannot inspect truth.
    final["final_class"] = "Low"
    assert validate_final(final, payload)[0] == final


def test_incomplete_or_refused_response_never_counts_as_valid_class():
    completed = {"status": "completed", "output": [{"type": "message", "status": "completed", "content": [
        {"type": "output_text", "text": json.dumps(native_final(), ensure_ascii=False)}]}]}
    assert parse_completed_response(completed, {})[0] == native_final()
    incomplete = copy.deepcopy(completed)
    incomplete["status"] = "incomplete"
    with pytest.raises(ValueError, match="Incomplete"):
        parse_completed_response(incomplete, {})
    refused = copy.deepcopy(completed)
    refused["output"][0]["content"] = [{"type": "refusal", "refusal": "declined"}]
    with pytest.raises(ValueError, match="refused"):
        parse_completed_response(refused, {})


def test_midrank_ties_and_out_of_range_scores():
    import numpy as np
    refs = np.asarray([[1., 2.], [1., 3.], [4., 3.]])
    assert _percentiles(refs, [[1., 3.], [-100., 100.]]).tolist() == [[1 / 3, 2 / 3], [0., 1.]]
