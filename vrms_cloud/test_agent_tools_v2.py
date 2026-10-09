import unittest
import numpy as np
from .agent_tools_v2 import FilterBankCSP, compact_spectrum, filterbank_covariance


class AdditionalToolContracts(unittest.TestCase):
    def test_filterbank_features_do_not_change_with_global_gain(self):
        rng=np.random.default_rng(2026)
        x=rng.normal(size=(2,30,1280))
        a=filterbank_covariance(x);b=filterbank_covariance(x*7)
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_csp_fit_never_reads_query_targets(self):
        rng=np.random.default_rng(2026)
        cov=[]
        for _ in range(12):
            x=rng.normal(size=(4,30,40));cov.append(np.einsum('bct,bdt->bcd',x,x).ravel())
        x=np.asarray(cov);y=np.tile([0,1],6)
        a=FilterBankCSP().fit(x[:8],y[:8]).transform(x[8:])
        y[8:]=1-y[8:]
        b=FilterBankCSP().fit(x[:8],y[:8]).transform(x[8:])
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(4,16))

    def test_missing_reference_remains_an_explicit_feature(self):
        rng=np.random.default_rng(2026)
        a=rng.normal(size=(20,240))
        x=compact_spectrum(a)
        self.assertEqual(x[-1],0)
        np.testing.assert_array_equal(x[-25:-1],np.zeros(24))
        self.assertTrue(np.isfinite(x).all())


if __name__=='__main__':
    unittest.main()
