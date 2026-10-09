"""A seed repeat must change training randomness, while retaining isolation."""
import unittest
import numpy as np
import torch

from vrms_pilot import model_ablation_v3 as mil
from .seed_retest import set_training_seed
from . import agent_tools_v1


class SeedRetestTests(unittest.TestCase):
    def tearDown(self):
        set_training_seed(2026)

    def test_seed_changes_fit_and_same_seed_reproduces_it(self):
        x = np.random.default_rng(42).normal(size=(16,8)).astype(np.float32)
        paths = [dict(subject_key=i+1,label=i%2,accepted_windows=2,window_start=2*i,window_end=2*i+2) for i in range(8)]
        base = np.arange(6)
        set_training_seed(2026)
        _, a = mil.fit_mil(x,paths,base,torch.device("cpu"),epochs=2)
        set_training_seed(2027)
        _, b = mil.fit_mil(x,paths,base,torch.device("cpu"),epochs=2)
        _, repeat = mil.fit_mil(x,paths,base,torch.device("cpu"),epochs=2)
        self.assertNotEqual(a["initial_state_sha256"],b["initial_state_sha256"])
        self.assertNotEqual(a["final_state_sha256"],b["final_state_sha256"])
        self.assertEqual(b["final_state_sha256"],repeat["final_state_sha256"])
        changed = [dict(p) for p in paths]
        for p in changed[6:]:
            p["label"] = 1-p["label"]
        _, poisoned = mil.fit_mil(x,changed,base,torch.device("cpu"),epochs=2)
        self.assertEqual(b["final_state_sha256"],poisoned["final_state_sha256"])

    def test_auxiliary_estimators_receive_new_seed(self):
        set_training_seed(2027)
        for config in ("lr_0.1","trees"):
            self.assertEqual(agent_tools_v1.classifier(config).steps[-1][1].random_state,2027)


if __name__ == "__main__":
    unittest.main()
