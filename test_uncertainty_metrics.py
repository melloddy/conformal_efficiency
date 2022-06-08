import unittest
import numpy as np
from scipy.sparse import csr_matrix
import ce.uncertainty_metrics as um


class TestUncertaintyMetrics(unittest.TestCase):
    def test_um_var_equal(self):
        arr = np.array([[[0.5, 0.5, 0.5]], [[0.75, 0.75, 0.75]]])
        var = um.calcVar(arr)
        assert var[0] == 0
        assert var[1] == 0

    def test_um_var_spread(self):
        arr = np.array([[[0, 0.5, 0.5, 1]], [[0, 0, 1, 1]]])
        var = um.calcVar(arr)
        assert np.allclose(var[0], np.var(arr[0, :]))
        assert np.allclose(var[1], np.var(arr[1, :]))

    def test_um_ig_equal(self):
        arr = np.array([[[0.5, 0.5, 0.5]], [[0.75, 0.75, 0.75]]])
        ig = um.calcVar(arr)
        assert ig[0] == 0
        assert ig[1] == 0

    def test_um_ig_spread(self):
        arr = np.array([[[0, 0.5, 0.5, 1]], [[0, 0, 1, 1]]])
        ig = um.calcIG(arr)
        assert ig[0] == 0.5
        assert ig[1] == 1
