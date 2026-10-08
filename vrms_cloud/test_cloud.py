"""Test blinded prompts and failure-sensitive parsing, not implementation mirrors."""
import json
import unittest

from vrms_cloud.cloud import build_job, check_blind, parse_predictions


class CloudContracts(unittest.TestCase):
    def test_query_label_changes_cannot_change_prompt(self):
        fold = dict(outer_subject=9, records=[dict(path_index=0, role="meta", evidence={"p": .4}),
                                            dict(path_index=1, role="outer_test", evidence={"p": .6}),
                                            dict(path_index=2, role="policy_validation", evidence={"p": .5})])
        evaluation = {i: dict(label=i % 2, subject_key=i + 1) for i in range(3)}
        first = build_job(fold, evaluation, "few_shot")
        evaluation[1]["label"] ^= 1
        evaluation[2]["label"] ^= 1
        second = build_job(fold, evaluation, "few_shot")
        self.assertEqual(first["user_message"], second["user_message"])
        self.assertEqual(first["example_indices"], [0])

    def test_private_fields_cannot_enter_nested_evidence(self):
        for key in ("label", "subject_key", "path_start_sec", "ssq"):
            with self.assertRaises(ValueError):
                check_blind({"nested": [{key: 0}]})

    def response(self, predictions, status="completed"):
        return dict(status=status, output=[dict(content=[dict(type="output_text", text=json.dumps(dict(predictions=predictions)))])])

    def prediction(self, ident="a", p=.7):
        return dict(id=ident, high_probability=p, state="high", uncertain=True, reason="conflicting evidence")

    def test_partial_duplicate_or_invalid_response_is_failure(self):
        cases = [[self.prediction()], [self.prediction(), self.prediction()],
                 [self.prediction("a", 1.2), self.prediction("b")],
                 [self.prediction("a", .2), self.prediction("b")]]
        for case in cases:
            with self.assertRaises(ValueError):
                parse_predictions(self.response(case), ["a", "b"])

    def test_incomplete_response_cannot_count_as_cloud_result(self):
        with self.assertRaises(ValueError):
            parse_predictions(self.response([self.prediction()], "incomplete"), ["a"])
        self.assertEqual(len(parse_predictions(self.response([self.prediction()]), ["a"])), 1)


if __name__ == "__main__":
    unittest.main()
