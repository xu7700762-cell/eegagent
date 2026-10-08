"""Synthetic end-to-end V2 contracts; random CNN weights are never trained."""
import copy
from functools import partial
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from vrms_cloud.evaluate import evaluate_v2
from vrms_cloud.cloud import SCHEMA, SYSTEM
from vrms_cloud.independent import load_prepared, run_independent
from vrms_cloud.prepare import prepare_v2
from vrms_cloud.validate import validate_v2
from vrms_deepseek.supervisor import assess_evidence as assess_deepseek
from vrms_pilot.data import digest, write_json
from vrms_pilot.model import CompactEEGCNN, predict_windows
from vrms_pilot.reliability import fit_reliability, signal_descriptor


class V2Pipeline(unittest.TestCase):
    def fixture(self, directory):
        """Tiny learned internal heads, with four explicit acceptance scenarios."""
        pilot = Path(directory) / "pilot"
        cache, results = pilot / "cache", pilot / "results"
        checkpoints = results / "checkpoints"
        cache.mkdir(parents=True)
        checkpoints.mkdir(parents=True)
        rng = np.random.default_rng(876)
        windows = rng.normal(0., 10., (6, 30, 1280)).astype(np.float32)
        np.save(cache / "windows.npy", windows)
        references = np.repeat((100. * np.eye(30))[None], 4, axis=0)
        np.save(cache / "path_references.npy", references)
        np.save(cache / "path_reference_power.npy", np.ones((4, 30, 4)))
        paths, cursor = [], 0
        for index in range(4):
            count = 0 if index == 2 else 2
            paths.append(dict(path_index=index, subject_key=index + 1, label=index % 2,
                episode_handle=f"synthetic-{index}", path_end_sec=10., accepted_windows=count,
                total_possible_windows=2, window_start=cursor, window_end=cursor + count,
                reference_available=True))
            cursor += count
        write_json(cache / "evaluation_manifest.json", paths)
        write_json(cache / "eligibility_audit.json", dict(synthetic=True, paths=4))
        write_json(cache / "dataset_audit.json", dict(schema_version="raw_event_eligibility_v2",
            paths=4, subjects=4, accepted_windows=6,
            eligibility_sha256=digest(cache / "eligibility_audit.json"), synthetic=True))
        (cache / "window_evidence.jsonl").write_text("".join(
            json.dumps(dict(start_sample=(i % 2) * 5120)) + "\n" for i in range(6)), encoding="utf-8")
        splits = [dict(outer_subject=i + 1, base_train=[10, 11], meta_calibration=[20, 21, 22],
            head_fit=[22], probability_calibration=[20, 21], policy_validation=[30, 31],
            outer_test=[i + 1]) for i in range(4)]
        write_json(pilot / "split_manifest.json", splits)
        torch.manual_seed(87)
        torch.set_num_threads(2)
        cnn = CompactEEGCNN().eval()
        mean, scale = torch.zeros((1, 30, 1)), torch.full((1, 30, 1), 10.)
        z = predict_windows(cnn, mean, scale, windows[:2], torch.device("cpu")).mean()
        # A real small head fit on synthetic features gives this test path a
        # clear high score. No EEG training or model-performance claim is made.
        head = LogisticRegression(C=100., solver="liblinear", random_state=87).fit(
            np.asarray([z - 4., z - 3., z - 2., z - 1.])[:, None], [0, 0, 1, 1])
        descriptor = signal_descriptor(windows[:2])
        groups = np.repeat([10, 11, 22, 20, 21, 30, 31], 10)
        labels = np.tile([0, 1], 35)
        probabilities = np.where(labels == 1, .95, .05)
        descriptors = descriptor + rng.normal(0., .05, (70, len(descriptor)))
        reliable = fit_reliability(probabilities, labels, groups, descriptors, np.ones(70),
            base_indices=np.flatnonzero(np.isin(groups, [10, 11])),
            head_indices=np.flatnonzero(groups == 22),
            calibration_indices=np.flatnonzero(np.isin(groups, [20, 21])),
            policy_indices=np.flatnonzero(np.isin(groups, [30, 31])))
        self.assertIsNotNone(reliable.calibrator)
        self.assertIsNotNone(reliable.thresholds)
        self.assertTrue(reliable.assess(float(head.predict_proba([[z]])[0, 1]),
                                       descriptor, 1.)["prediction_reliable"])
        for index, split in enumerate(splits):
            gate = copy.deepcopy(reliable)
            if index in (1, 3):
                gate.thresholds = None
            if index == 3:
                gate.calibrator = None
            stem = checkpoints / f"outer_subject_{index + 1:02d}"
            torch.save(dict(model=cnn.state_dict(), mean=mean, scale=scale,
                            training_subjects=split["base_train"]), str(stem) + ".pt")
            joblib.dump(dict(split=split, reliability=gate,
                meta=dict(deep=head, deep_temporal=None, biomarker=None, covariance=None),
                classifiers=dict(bio_absolute=None, bio_reference=None, covariance=None)),
                str(stem) + ".joblib")
        for name in ("path_oof.csv", "inner_predictions.csv"):
            (results / name).write_text("synthetic_fixture_only\n", encoding="utf-8")
        write_json(results / "protocol.json", dict(schema_version="pilot_v2", smoke=True, synthetic=True,
            external_code_sha256={"vrms_refine/retrieval.py": digest(Path("vrms_refine/retrieval.py"))},
            code_sha256={name: digest(Path("vrms_pilot") / name) for name in
                         ("model.py", "engine.py", "reliability.py", "experiment.py")}))
        return pilot

    def prepare(self, out, pilot):
        # If preparation touches either expensive physiological operation,
        # failure identifies eager execution before the Supervisor can decide.
        with redirect_stdout(io.StringIO()), \
             patch("vrms_pilot.experiment.spectral_features", side_effect=AssertionError("eager biomarker")), \
             patch("vrms_pilot.experiment.covariance_features", side_effect=AssertionError("eager covariance")):
            return prepare_v2(out, pilot)

    def test_offline_prepare_replay_evaluate_preserves_abstention_and_lazy_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = self.fixture(tmp)
            out = Path(tmp) / "cloud"
            protocol = self.prepare(out, pilot)
            self.assertTrue(protocol["partial_run"])
            self.assertFalse(protocol["llm_changes_probability"])
            folds = json.loads((out / "fold_evidence.json").read_text(encoding="utf-8"))
            for fold in folds:
                for record in fold["records"]:
                    self.assertNotIn("physiological_evidence", record["evidence"])
                    self.assertNotIn("case_retrieval", record["evidence"])
            assessor = Mock(side_effect=AssertionError("offline replay called an API assessor"))
            with redirect_stdout(io.StringIO()), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("offline network call")):
                status = run_independent(out, pilot, assessor=assessor, workers=2, dry_run=True)
                summary = evaluate_v2(out, offline=True)
            assessor.assert_not_called()
            self.assertEqual(status["cloud_calls"], 0)
            self.assertEqual(summary["state_counts"], dict(high=1, uncertain=2, insufficient_data=1))
            self.assertEqual(summary["model"]["published_paths"], 1)
            self.assertEqual(summary["model"]["scoreable_paths"], 2)
            self.assertEqual(summary["model"]["publication_coverage"], .25)
            packs = [json.loads((out / "offline_calls" / f"path_{i:03d}_evidence_pack.json")
                     .read_text(encoding="utf-8")) for i in range(4)]
            self.assertEqual(set(packs[0]["tool_calls"]),
                             {"SignalQualityChecker", "DeepVRMSDetector", "UncertaintyEvaluator"})
            self.assertIsNone(packs[0]["physiological_evidence"])
            for index in (1, 3):
                self.assertTrue({"CaseRetriever", "BiomarkerCalculator", "CovarianceAnalyzer"}
                                .issubset(packs[index]["tool_calls"]))
                self.assertEqual(packs[index]["physiological_evidence"]["verification"]["status"], "inconclusive")
                self.assertFalse(packs[index]["physiological_evidence"]["verification"]["direction_validated"])
            self.assertNotIn("DeepVRMSDetector", packs[2]["tool_calls"])
            self.assertIsNone(packs[2]["high_probability"])
            self.assertIsNone(packs[3]["high_probability"])
            self.assertIsNotNone(packs[3]["p_raw"])
            for pack in packs:
                self.assertEqual(pack["high_probability"], pack["p_cal"])
                self.assertFalse(pack["cloud_called"])
                self.assertFalse(pack["fallback_used"])
                self.assertEqual(pack["effective_llm_weight"], 0)

    def test_frozen_checkpoint_or_prepared_evidence_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = self.fixture(tmp)
            out = Path(tmp) / "cloud"
            self.prepare(out, pilot)
            for filename, message in ((pilot / "results/checkpoints/outer_subject_01.pt", "checkpoint"),
                                      (out / "fold_evidence.json", "artifact")):
                original = filename.read_bytes()
                filename.write_bytes(original + b"\n")
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        load_prepared(out, pilot)
                finally:
                    filename.write_bytes(original)

    def test_frozen_eeg_reference_and_timing_cache_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = self.fixture(tmp)
            out = Path(tmp) / "cloud"
            self.prepare(out, pilot)
            for name in ("windows.npy", "path_reference_power.npy", "window_evidence.jsonl"):
                filename = pilot / "cache" / name
                original = filename.read_bytes()
                if name.endswith(".npy"):
                    values = np.load(filename)
                    values.flat[0] += 1.
                    np.save(filename, values)
                else:
                    rows = [json.loads(line) for line in original.decode("utf-8").splitlines()]
                    rows[0]["start_sample"] += 1024
                    filename.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                try:
                    with self.assertRaisesRegex(ValueError, "Frozen cache artifact changed"):
                        load_prepared(out, pilot)
                finally:
                    filename.write_bytes(original)

    def test_cached_vendor_explanations_validate_and_final_pack_tampering_is_rejected(self):
        prediction = dict(id="qsingle", supporting_evidence=["synthetic deep model evidence"],
                          conflicting_evidence=[], missing_evidence=["independently validated exit"],
                          explanation="Synthetic response explains the machine abstention")
        for vendor in ("gpt", "deepseek"):
            with self.subTest(vendor=vendor), tempfile.TemporaryDirectory() as tmp:
                pilot = self.fixture(tmp)
                out = Path(tmp) / vendor
                self.prepare(out, pilot)

                def fake_responses(provider, job, destination):
                    raw = dict(status="completed", output=[dict(content=[dict(type="output_text",
                        text=json.dumps(dict(predictions=[prediction])))])])
                    body = dict(input=[dict(role="system", content=SYSTEM),
                                       dict(role="user", content=job["user_message"])],
                                text=dict(format=dict(type="json_schema", schema=SCHEMA, strict=True)))
                    log = dict(**job, provider="configured_responses", body=body,
                               predictions=[prediction], success=True, seconds=0.,
                               attempts=[dict(response=raw, status="success")])
                    write_json(destination, log)
                    return log

                captured = SimpleNamespace(response=None)
                def fake_completions(*args, **kwargs):
                    reply = dict(predictions=[prediction])
                    captured.response = dict(model="synthetic-deepseek", choices=[dict(finish_reason="stop",
                        message=dict(content=json.dumps(reply)))])
                    return reply
                provider = SimpleNamespace(model="synthetic-deepseek", thinking_disabled=True,
                                           call=Mock(side_effect=fake_completions))
                assessor = (partial(assess_deepseek, provider, captured, dict(base_url="https://api.deepseek.com"))
                            if vendor == "deepseek" else None)
                with redirect_stdout(io.StringIO()), \
                     patch("urllib.request.urlopen", side_effect=AssertionError("synthetic test made a network call")), \
                     patch("vrms_cloud.supervisor.existing_provider", return_value={}), \
                     patch("vrms_cloud.supervisor.call_job", side_effect=fake_responses) as responses:
                    args = {} if assessor is None else dict(assessor=assessor)
                    status = run_independent(out, pilot, workers=1, **args)
                    validation = validate_v2(out)
                self.assertEqual(status["cloud_calls"], 2)
                self.assertEqual(validation["independent_requests_checked"], 2)
                self.assertEqual(validation["status"], "passed")
                self.assertEqual(provider.call.call_count if vendor == "deepseek" else responses.call_count, 2)
                path = out / "independent_calls/path_001_evidence_pack.json"
                original = path.read_bytes()
                mutations = [dict(explanation="text absent from the actual response"),
                    dict(supporting_evidence=["unsupported added evidence"]), dict(cloud_success=False),
                    dict(fallback_used=True), dict(cloud_called=False), dict(p_cal=.51, high_probability=.51)]
                for mutation in mutations:
                    with self.subTest(vendor=vendor, changed_fields=list(mutation)):
                        pack = json.loads(original.decode("utf-8"))
                        pack.update(mutation)
                        write_json(path, pack)
                        try:
                            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                                validate_v2(out)
                        finally:
                            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
