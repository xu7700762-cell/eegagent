"""Leakage and evaluation regressions for the numeric-only V3 experiment."""
import unittest

import numpy as np

from .engine_v3 import reliability_decision
from .reliability_experiment_v3 import crossfit_meta_head
from .risk_coverage_v3 import risk_curve, score_metrics


class V3ExperimentContracts(unittest.TestCase):
    def test_oof_head_never_fits_its_query_subject_label(self):
        subjects=np.repeat(np.arange(1,7),8)
        y=np.tile([0,1],24)
        x=(y*.8+np.linspace(-.1,.1,len(y)))[:,None]
        original,lineage=crossfit_meta_head(x,y,subjects,[1,2,3,4,5])
        changed=y.copy()
        changed[subjects==3]=1-changed[subjects==3]
        rescored,_=crossfit_meta_head(x,changed,subjects,[1,2,3,4,5])
        np.testing.assert_array_equal(original[subjects==3],rescored[subjects==3])
        self.assertTrue(np.isnan(original[subjects==6]).all())
        for row in lineage:
            self.assertNotIn(row['heldout_subject'],row['head_fit_subjects'])

    def test_outer_data_changes_cannot_fit_meta_heads(self):
        subjects=np.repeat(np.arange(1,7),8)
        y=np.tile([0,1],24)
        x=(y*.8+np.linspace(-.1,.1,len(y)))[:,None]
        original,_=crossfit_meta_head(x,y,subjects,[1,2,3,4,5])
        x[subjects==6]=1e5
        y[subjects==6]=1
        changed,_=crossfit_meta_head(x,y,subjects,[1,2,3,4,5])
        np.testing.assert_array_equal(original,changed)

    def test_coverage_warning_is_uncertain_not_insufficient(self):
        decision=reliability_decision(dict(p_cal=.8,prediction_reliable=False,signal_quality_bad=False,
            policy_status='validated',ood=False,rejection_reasons=['coverage_warning']))
        self.assertEqual(decision['state'],'uncertain')
        self.assertEqual(decision['probability'],.8)

    def test_unknown_ood_or_missing_calibration_cannot_release(self):
        common=dict(prediction_reliable=True,signal_quality_bad=False,policy_status='validated',rejection_reasons=[])
        self.assertEqual(reliability_decision(dict(common,p_cal=.99,ood=None))['state'],'uncertain')
        self.assertEqual(reliability_decision(dict(common,p_cal=None,ood=False))['state'],'uncertain')

    def test_risk_ranking_preserves_ties_and_ignores_truth(self):
        p=np.array([.1,.9,.4,.6,np.nan])
        y=np.array([0,1,0,1,1])
        a=risk_curve(y,p,[1,1,2,2,3])
        b=risk_curve(1-y,p,[1,1,2,2,3])
        self.assertEqual(a['rows'][0]['accepted_paths'],2)
        self.assertEqual(a['rows'][0]['coverage'],.4)
        self.assertEqual(a['rows'][-1]['coverage'],.8)
        self.assertEqual([r['margin'] for r in a['rows']],[r['margin'] for r in b['rows']])
        self.assertEqual([r['accepted_paths'] for r in a['rows']],[r['accepted_paths'] for r in b['rows']])

    def test_single_class_fold_auroc_is_null_brier_is_retained(self):
        score=score_metrics([1,1],[.8,.9])
        self.assertIsNone(score['auroc'])
        self.assertIsNone(score['bacc'])
        self.assertAlmostEqual(score['brier'],.025)
        self.assertIsNotNone(score['ece'])

    def test_no_score_fold_has_no_invented_zero_risk_or_auroc(self):
        score=score_metrics([],[])
        self.assertIsNone(score['score_coverage'])
        self.assertIsNone(score['brier'])
        self.assertEqual(risk_curve([0,1],[np.nan,np.nan],[1,2])['rows'],[])


if __name__=='__main__':
    unittest.main()
