import unittest
import numpy as np
from scipy.sparse import csr_matrix
import ce.cp_and_calibration as cp


class TestCP(unittest.TestCase):
    eps = 0.05
    population_size = 100

    def test_cp_separate_distributions(self):
        """
        Separated distribution for fitting CP will lead to clean split and no overlap in the centre
        """
        np.random.seed(42)
        neg_preds = np.clip(np.random.normal(0.2, 0.1, self.population_size), 0.0001, 1)
        pos_preds = np.clip(np.random.normal(0.8, 0.1, self.population_size), 0.0001, 1)
        neg_labels = np.array([-1] * self.population_size)
        pos_labels = np.array([1] * self.population_size)
        preds_fva_dict = csr_matrix(np.hstack([pos_preds, neg_preds])).T
        labels_fva_dict = csr_matrix(np.hstack([pos_labels, neg_labels])).T
        ncms_fva_fit_dict, labels_fva_fit_dict = cp.fit_cp(
            preds_fva_dict, labels_fva_dict
        )
        neg_preds_fte = np.clip(
            np.random.normal(0.05, 0.05, self.population_size), 0.0001, 1
        )
        pos_preds_fte = np.clip(
            np.random.normal(0.95, 0.05, self.population_size), 0.0001, 1
        )

        # unambiguous predictions act/inact
        cp_test_neg, _, _ = cp.apply_cp(
            neg_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 1 not in cp_test_neg[0]
        assert "uncertain both" not in cp_test_neg[0]
        assert "uncertain none" not in cp_test_neg[0]

        cp_test_pos, _, _ = cp.apply_cp(
            pos_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 0 not in cp_test_pos[0]
        assert "uncertain both" not in cp_test_pos[0]
        assert "uncertain none" not in cp_test_pos[0]

        # ambiguous predictions will get no label since it falls outside the confidence ranges for both classses

        cp_test_center, _, _ = cp.apply_cp(
            np.array([0.5])[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert "uncertain none" in cp_test_center[0]

        # uncertain predictions from calibration set will get no labels when NCV is too extreme

        # neg
        cp_test_neg, _, _ = cp.apply_cp(
            neg_preds[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert (
            len([e for e in cp_test_neg[0] if e == 0]) / len(cp_test_neg[0])
            >= 1 - self.eps
        )
        assert "uncertain both" not in cp_test_neg

        n_non_neg = len(cp_test_neg[0]) - len([e for e in cp_test_neg[0] if e == 1])
        assert 1 not in list(
            np.array(cp_test_neg[0])[(np.argsort(neg_preds)[:n_non_neg])]
        )
        assert 0 not in list(
            np.array(cp_test_neg[0])[(np.argsort(neg_preds)[:n_non_neg])]
        )
        assert "uncertain both" not in list(
            np.array(cp_test_neg[0])[(np.argsort(neg_preds)[:n_non_neg])]
        )

        # pos

        cp_test_pos, _, _ = cp.apply_cp(
            pos_preds[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert (
            len([e for e in cp_test_pos[0] if e == 1]) / len(cp_test_pos[0])
            >= 1 - self.eps
        )
        assert "uncertain both" not in cp_test_pos

        n_non_pos = len(cp_test_pos[0]) - len([e for e in cp_test_pos[0] if e == 1])
        assert 1 not in list(
            np.array(cp_test_pos[0])[(np.argsort(pos_preds)[:n_non_pos])]
        )
        assert 0 not in list(
            np.array(cp_test_pos[0])[(np.argsort(pos_preds)[:n_non_pos])]
        )
        assert "uncertain both" not in list(
            np.array(cp_test_pos[0])[(np.argsort(pos_preds)[:n_non_pos])]
        )

    def test_cp_identical_distributions(self):
        """
        Identical distribution for fitting CP will lead to no split but still certainty away from the distribution
        Negative side
        """
        np.random.seed(42)
        neg_preds = np.clip(np.random.normal(0.2, 0.1, self.population_size), 0.0001, 1)
        pos_preds = np.clip(np.random.normal(0.8, 0.1, self.population_size), 0.0001, 1)
        neg_labels = np.array([-1] * self.population_size)
        pos_labels = np.array([1] * self.population_size)

        preds_fva_dict = csr_matrix(np.hstack([neg_preds, neg_preds])).T
        labels_fva_dict = csr_matrix(np.hstack([neg_labels, neg_labels])).T
        ncms_fva_fit_dict, labels_fva_fit_dict = cp.fit_cp(
            preds_fva_dict, labels_fva_dict
        )

        neg_preds_fte = np.clip(
            np.random.normal(0.05, 0.05, self.population_size), 0.0001, 1
        )
        pos_preds_fte = np.clip(
            np.random.normal(0.95, 0.05, self.population_size), 0.0001, 1
        )

        cp_test_neg, _, _ = cp.apply_cp(
            neg_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 1 not in cp_test_neg[0]
        assert 0 not in cp_test_neg[0]
        assert "uncertain none" not in cp_test_neg[0]

        cp_test_pos, _, _ = cp.apply_cp(
            pos_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 0 not in cp_test_pos[0]
        assert "uncertain both" not in cp_test_pos[0]
        assert "uncertain none" not in cp_test_pos[0]

        """
        Identical distribution for fitting CP will lead to no split but still certainty away from the distribution
        Positive side 
        """

        preds_fva_dict = csr_matrix(np.hstack([pos_preds, pos_preds])).T
        labels_fva_dict = csr_matrix(np.hstack([pos_labels, pos_labels])).T
        ncms_fva_fit_dict, labels_fva_fit_dict = cp.fit_cp(
            preds_fva_dict, labels_fva_dict
        )

        cp_test_neg, _, _ = cp.apply_cp(
            neg_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 1 not in cp_test_neg[0]
        assert "uncertain both" not in cp_test_neg[0]
        assert "uncertain none" not in cp_test_neg[0]

        cp_test_pos, _, _ = cp.apply_cp(
            pos_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 1 not in cp_test_pos[0]
        assert 0 not in cp_test_pos[0]
        assert "uncertain none" not in cp_test_pos[0]

    def test_cp_overlapping_distributions(self):
        """
        Overlapping distribution for fitting CP will lead to overlap in the centre where both labels will be assigned
        """

        np.random.seed(42)
        neg_preds = np.clip(np.random.normal(0.4, 0.1, self.population_size), 0.0001, 1)
        pos_preds = np.clip(np.random.normal(0.6, 0.1, self.population_size), 0.0001, 1)

        neg_labels = np.array([-1] * self.population_size)
        pos_labels = np.array([1] * self.population_size)

        neg_preds_fte = np.clip(
            np.random.normal(0.05, 0.05, self.population_size), 0.0001, 1
        )
        pos_preds_fte = np.clip(
            np.random.normal(0.95, 0.05, self.population_size), 0.0001, 1
        )

        preds_fva_dict = csr_matrix(np.hstack([pos_preds, neg_preds])).T
        labels_fva_dict = csr_matrix(np.hstack([pos_labels, neg_labels])).T
        ncms_fva_fit_dict, labels_fva_fit_dict = cp.fit_cp(
            preds_fva_dict, labels_fva_dict
        )

        cp_test_neg, _, _ = cp.apply_cp(
            neg_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )

        cp_test_neg, _, _ = cp.apply_cp(
            neg_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 1 not in cp_test_neg[0]
        assert "uncertain both" not in cp_test_neg[0]
        assert "uncertain none" not in cp_test_neg[0]

        cp_test_pos, _, _ = cp.apply_cp(
            pos_preds_fte[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert 0 not in cp_test_pos[0]
        assert "uncertain both" not in cp_test_neg[0]
        assert "uncertain none" not in cp_test_pos[0]

        cp_test_center, _, _ = cp.apply_cp(
            np.array([0.5])[:, None], labels_fva_fit_dict, ncms_fva_fit_dict, self.eps
        )
        assert "uncertain both" in cp_test_center[0]
