"""Portable numeric replay and six-field response checks for the RAG experiment.

No network requests or signal inference run in this module. Public classification
labels are kept separately from inputs and are used only by the offline scorer.
"""
import copy
import hashlib
import json
from pathlib import Path

from .decisive_rag import FAMILIES, _values, context_from_bank, validate_bank

RESULTS = Path(__file__).resolve().parents[1] / "results"
FINAL_FIELDS = {"final_class", "confidence", "explanation", "supporting_tools", "counterevidence_tools", "rag_citations"}
HIDDEN_KEYS = {"true_class", "questionnaire_score", "questionnaire_answers", "previous_answer", "cloud_judgment",
               "raw_combined_class", "raw_combined_score", "predicted_class"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hidden(value):
    if isinstance(value, dict):
        if HIDDEN_KEYS & set(value):
            raise ValueError("Query label, previous answer or fusion output in payload")
        for nested in value.values():
            check_hidden(nested)
    elif isinstance(value, list):
        for nested in value:
            check_hidden(nested)


def load_replay(directory=RESULTS):
    """Validate frozen public bytes before accepting numerical evidence."""
    directory = Path(directory)
    manifest = read(directory / "optimized_rag_manifest.json")
    for name, expected in manifest["exported_files_sha256"].items():
        # Only flat result filenames from this exact replay manifest are accepted.
        if not name.startswith("results/optimized_rag_") or len(Path(name).parts) != 2:
            raise ValueError("Invalid replay file name")
        if hashlib.sha256((directory / Path(name).name).read_bytes()).hexdigest() != expected:
            raise ValueError("Replay file changed: " + name)
    inputs = read(directory / "optimized_rag_inputs.json")
    banks = read(directory / "optimized_rag_banks.json")
    protocol = read(directory / "optimized_rag_protocol.json")
    expected_ids = {f"q{i:03d}" for i in range(1, 147)}
    if set(inputs) != expected_ids or set(manifest["contexts"]) != expected_ids or protocol["fixed_total"] != 146:
        raise ValueError("Incomplete fixed replay query set")
    for fold, entry in banks.items():
        validate_bank(entry["bank"], int(fold))
    for query in inputs.values():
        if str(query["fold"]) not in banks:
            raise ValueError("Missing query reference fold")
        payload = query["without_rag_payload"]
        check_hidden(payload)
        if "policy_heldout_case_context" in payload["initial_evidence"]:
            raise ValueError("Baseline already contains RAG")
        prediction = query["prediction"]
        tools = payload["corrective_evidence"]["actual_individual_evidence"]
        source = _values(prediction["mil_raw_high_probability"], prediction["numeric_fusion"]["actual_tools"])
        supplied = _values(payload["initial_evidence"]["baseline"]["raw_high_probability"], tools)
        if supplied != source or payload["corrective_evidence"]["reference_available"] is not False:
            raise ValueError("RAG query readings differ from baseline input")
    return inputs, banks, protocol, manifest


def payload_for(query_id, arm, inputs, banks, manifest):
    """Build either arm from one immutable baseline; no evaluation labels read."""
    if arm not in ("without_rag", "percentile_group"):
        raise ValueError("Unknown experiment arm")
    query = inputs[query_id]
    payload = copy.deepcopy(query["without_rag_payload"])
    if arm == "percentile_group":
        entry = banks[str(query["fold"])]
        context = context_from_bank(query["prediction"], entry["bank"], entry["historical_provenance"], "percentile_group")
        if digest(context) != manifest["contexts"][query_id]:
            raise ValueError("Context differs from transformed frozen API evidence: " + query_id)
        payload["initial_evidence"]["policy_heldout_case_context"] = context
    check_hidden(payload)
    return payload


def citation_ids(value):
    found = set()
    if isinstance(value, dict):
        found.update(value[key] for key in ("citation_id", "case_id") if isinstance(value.get(key), str))
        for nested in value.values():
            found.update(citation_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(citation_ids(nested))
    return found


def validate_final(value, payload):
    """Keep native six fields; citation names produce separate metadata warnings.

    Unknown references are not verified citations and are never grounds for
    resampling a completed classification. This applies without looking at truth.
    """
    if not isinstance(value, dict) or set(value) != FINAL_FIELDS:
        raise ValueError("Invalid six-field final")
    if value["final_class"] not in ("High", "Low") or value["confidence"] not in ("low", "medium", "high"):
        raise ValueError("Invalid final class or confidence")
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise ValueError("Missing explanation")
    for key in ("supporting_tools", "counterevidence_tools", "rag_citations"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) for item in value[key]):
            raise ValueError("Invalid evidence array")
    warnings = []
    for key in ("supporting_tools", "counterevidence_tools"):
        for name in value[key]:
            if name == "baseline":
                warnings.append({"field": key, "reference": name, "kind": "known_MIL_input_alias"})
            elif name not in {"MIL", *FAMILIES}:
                warnings.append({"field": key, "reference": name, "kind": "unknown_tool_reference"})
    ids = citation_ids(payload)
    for name in value["rag_citations"]:
        if name not in ids:
            warnings.append({"field": "rag_citations", "reference": name, "kind": "unknown_RAG_reference"})
    return copy.deepcopy(value), warnings


def parse_completed_response(response, payload):
    """Reject incomplete/refused output before accepting any binary judgment."""
    if response.get("status") != "completed":
        raise ValueError("Incomplete model response")
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("status") not in (None, "completed"):
            raise ValueError("Incomplete output message")
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise ValueError("Model refused to classify")
            if content.get("type") == "output_text":
                chunks.append(content["text"])
    return validate_final(json.loads("".join(chunks)), payload)
