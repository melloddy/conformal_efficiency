import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from scipy.sparse import csc_matrix, spmatrix
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
import numpy as np
import glob
import sys
import os
from typing import Tuple


def perform_platt_scaling(
    labels_va: spmatrix, preds_va: spmatrix, eligible_tasks:list
                         )->pd.DataFrame:

    coeffs = []
    intercepts = []
    idx = []

    for col in tqdm(sorted(eligible_tasks)):

        labels_va_col = labels_va[:, col].data

        preds_va_col = preds_va[:, col].data
        # ignoring the warnings for the divide by zero
        np.seterr(divide="ignore", invalid="ignore")
        logits_va_col = np.log(preds_va_col / (1 - preds_va_col))
        labels_va_col = labels_va_col[
            np.isfinite(logits_va_col)
        ]  # prevent inf that will crash the logistic regression
        logits_va_col = logits_va_col[
            np.isfinite(logits_va_col)
        ]  # prevent inf that will crash the logistic regression
        
        
        ## Fitting log reg
        try:
            model = LogisticRegression()
            clf = model.fit(logits_va_col.reshape(-1, 1), labels_va_col)
            coeffs.append(clf.coef_)
            intercepts.append(clf.intercept_)
            idx.append(col)
        except Exception as e:
            print(e)
            print("index failed: ", col)
            coeffs.append(np.array(np.nan))
            intercepts.append(np.array(np.nan))
            idx.append(col)

    df = pd.DataFrame(
        {
            "coeffs": [e.item() for e in coeffs],
            "intercepts": [e.item() for e in intercepts],
            "idx": idx,
        }
    )
    return df


def prob_ncm(scores, labels):
    """
    Returns nonconformity Measures for CP from model scores 
    Assumes that scores are directly related to the probability of being active
    """
    ncm = np.where(labels > 0, -scores, scores)
    return ncm 

def p_values(calibration_alphas, test_alphas, randomized=False):
    '''
    P-value calculation
    '''
    sorted_cal_alphas = sorted(calibration_alphas)
    if randomized:
        # for each test alpha, tieBreaker is the (number of calibration alphas with the same value)*(uniform RV between 0 and 1)
        tie_counts = np.searchsorted(
            sorted_cal_alphas, test_alphas, side="right"
        ) - np.searchsorted(sorted_cal_alphas, test_alphas)
        tie_breaker = (
            np.random.uniform(size=len(np.atleast_1d(test_alphas))) * tie_counts
        )
        return (
            len(calibration_alphas)
            - (
                np.searchsorted(sorted_cal_alphas, test_alphas, side="right")
                - tie_breaker
            )
            + 1
        ) / (len(calibration_alphas) + 1)
    else:
        return (
            len(calibration_alphas)
            - np.searchsorted(sorted_cal_alphas, test_alphas)
            + 1
        ) / (len(calibration_alphas) + 1)


def micp(
    calibration_alphas,
    calibration_labels,
    test_alphas_0,
    test_alphas_1,
    randomized=False,
):
    """
    Mondrian Inductive Conformal Predictor
    Parameters:
    calibration_alphas: 1d array of Nonconformity Measures for the calibration examples
    calibration_labels: 1d array of labels for the calibration examples - ideally 0/1 or -1/+1,
                        but negative/positive values also accepted
    test_alpha_0: 1d array of NCMs for the test examples, assuming 0 as label
    test_alpha_1: 1d array of NCMs for the test examples, assuming 1 as label
    Returns:
    p0,p1 : pair of arrays containing the p-values for label 0 and label 1
    """
    if not len(calibration_labels) == len(calibration_alphas):
        raise ValueError(
            "calibration_labels and calibration alphas must have the same size"
        )

    if not len(np.atleast_1d(test_alphas_0)) == len(np.atleast_1d(test_alphas_1)):
        raise ValueError("test_alphas_0 and test_alphas_1 must have the same size")

    p_0 = p_values(
        calibration_alphas[calibration_labels <= 0], test_alphas_0, randomized
    )
    p_1 = p_values(
        calibration_alphas[calibration_labels > 0], test_alphas_1, randomized
    )
    return p_0, p_1


def cp_label_predictor(p0, p1, eps):
    '''
    Punction to generate the label from p0 and p1
    Active: p1 > ϵ and p0 ≤ ϵ
    Inactive: p0 > ϵ and p1 ≤ ϵ
    Uncertain (Both): p1 > ϵ and p0 > ϵ
    Empty (None): p1 ≤ ϵ and p0 ≤ ϵ
    '''
    if p1 > eps and p0 <= eps:
        return 1
    elif p0 > eps and p1 <= eps:
        return 0
    elif p0 > eps and p1 > eps:
        return "uncertain both"
    elif p0 <= eps and p1 <= eps:
        # return 'empty'
        # it should actually return 'empty', but to avoid a confusion for people
        return "uncertain none"


def fit_cp(preds_fva: spmatrix, labels_fva: spmatrix) -> Tuple[dict, dict]:
    '''
    Generates the non-conformity values for a conformal predictor. 
    Input: Matching prediction and label sparse matrices. Each column corresponds to a task.
    Output: Non-conformity value and label dictionaries
    '''

    ncms_fva_fit_dict = {}
    labels_fva_fit_dict = {}

    preds_fva = csc_matrix(preds_fva)
    labels_fva = csc_matrix(labels_fva)

    for col in range(preds_fva.shape[1]):
        try:

            preds_fva_col = preds_fva.data[
                preds_fva.indptr[col] : preds_fva.indptr[col + 1]
            ]
            labels_fva_col = labels_fva.data[
                labels_fva.indptr[col] : labels_fva.indptr[col + 1]
            ]
            labels_fva_col = np.where(labels_fva_col == -1, 0, 1)

            ncms_fva = prob_ncm(preds_fva_col, labels_fva_col)
            ncms_fva_fit_dict[
                str(col)
            ] = (
                ncms_fva.tolist()
            )  # use tolist() to avoid difficulties with the serialisation
            labels_fva_fit_dict[
                str(col)
            ] = (
                labels_fva_col.tolist()
            )  # use tolist() to avoid difficulties with the serialisation
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(
                "Error encountered for index {}, during fit_cp function".format(
                    col
                )
            )
            print("Description is below")
            print(e)

    return ncms_fva_fit_dict, labels_fva_fit_dict


def calc_alpha_threshold(calibration_alphas, eps, approach='p_value'):
    '''
    Calculates the threshold in terms of predictive probabilities (alpha) for a single class 
    '''
    sorted_cal_alphas = sorted(calibration_alphas)
    if approach=='p_value': 
        pvals = p_values(calibration_alphas, calibration_alphas)
        if len(calibration_alphas[pvals<=eps])==0: 
            return np.nan
        else: 
            threshold=np.min(calibration_alphas[pvals<=eps]) # edge of the rejected values, least strange
    elif approach=='percentile':
        threshold = np.percentile(sorted_cal_alphas, (1-eps) * 100)
    return threshold


def apply_cp(
    preds_fte: spmatrix, labels_fva_dict: dict, ncms_fva_fit_dict: dict, eps: float
) -> Tuple[dict, dict, dict]:
    '''
    Generates prediction sets for a test set. 
    Input: predictive probabilities test set, labels calibration set, non-conformity values calibration set, error level
    Output: class label prediction sets, alpha values for both classes
    '''

    preds_fte = csc_matrix(preds_fte)

    cp_test = {}
    alpha_threshold_neg = {}
    alpha_threshold_pos = {}
    assert len(ncms_fva_fit_dict.keys()) == preds_fte.shape[1]

    for col in np.array(list(ncms_fva_fit_dict.keys()), dtype=int):
        try:

            ncms_fva_col = np.array(ncms_fva_fit_dict[str(col)])
            labels_fva_col = np.array(labels_fva_dict[str(col)])

            preds_fte_col = preds_fte.data[
                preds_fte.indptr[col] : preds_fte.indptr[col + 1]
            ]

            ncms_test_0 = prob_ncm(preds_fte_col, np.repeat(0.0, len(preds_fte_col)))
            ncms_test_1 = prob_ncm(preds_fte_col, np.repeat(1.0, len(preds_fte_col)))

            p0, p1 = micp(
                ncms_fva_col, labels_fva_col, ncms_test_0, ncms_test_1, randomized=False
            )

            alpha_threshold_neg[col] = calc_alpha_threshold(
                 ncms_fva_col[labels_fva_col <= 0], eps
            )
            alpha_threshold_pos[col] =  calc_alpha_threshold(
                ncms_fva_col[labels_fva_col > 0], eps
            )
            cp_test[col] = [
                cp_label_predictor(pe0, pe1, eps) for pe0, pe1 in zip(p0, p1)
            ]

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(
                "Error encountered for index {}, during apply_cp function".format(
                    col
                )
            )
            print("Description is below")
            print(e)

    return cp_test, alpha_threshold_neg, alpha_threshold_pos


def eval_cp(cp_test: dict, labels_fte_dict: dict) -> pd.DataFrame:
    '''
    Calculates evaluation metrics for a conformal predictor. 
    Input: class label prediction sets for the test set. Labels for the test set (optional). Both dictionaries with each key corresponding to a task
    Output: dataframe with evaluation metrics 
    '''

    idxs = []
    unis = []
    e_overall = []
    e_inacts = []
    e_acts = []
    e_inacts_nonone = []
    e_acts_nonone = []
    certain_acts = []
    certain_inacts = []
    uncertain_boths = []
    uncertain_nones = []
    # labelled
    val_inacts = []
    val_acts = []
    lit_val_inacts = []
    lit_val_acts = []
    n_acts = []
    n_inacts = []

    labelled_feval = True if len(labels_fte_dict.keys()) > 0 else False

    for col in cp_test.keys():

        cp_test_arr_col_str = np.array(cp_test[col], dtype=str)

        if labelled_feval:
            labels_fte_col = np.array(labels_fte_dict[str(col)])

        try:
            uni = np.unique(cp_test_arr_col_str)
            # dtype str is needed : in case cp_test only contains numeric types,
            # a FutureWarning will be raised and a scalar False/True will be returned
            idx_certain_inact = np.where(cp_test_arr_col_str == "0")[0]
            idx_certain_act = np.where(cp_test_arr_col_str == "1")[0]
            idx_uncertain_none = np.where(
                [e == "uncertain none" for e in cp_test_arr_col_str]
            )[0]
            idx_uncertain_both = np.where(
                [e == "uncertain both" for e in cp_test_arr_col_str]
            )[0]

            # efficiency
            efficiency_overall = (len(idx_certain_inact) + len(idx_certain_act)) / len(
                cp_test_arr_col_str
            )
            efficiency_act = len(idx_certain_act) / len(cp_test_arr_col_str)
            efficiency_inact = len(idx_certain_inact) / len(cp_test_arr_col_str)
            try:
                efficiency_act_nonone = len(idx_certain_act) / (
                    len(idx_certain_act) + len(idx_uncertain_both) + len(idx_certain_inact)
                )
            except ZeroDivisionError:
                efficiency_act_nonone = np.nan
            try:
                efficiency_inact_nonone = len(idx_certain_inact) / (
                    len(idx_certain_inact) + len(idx_uncertain_both) + len(idx_certain_act)
                )
            except ZeroDivisionError:
                efficiency_inact_nonone = np.nan

            if labelled_feval:
                idx_inact_labels = np.where(labels_fte_col == 0)[0]
                idx_act_labels = np.where(labels_fte_col == 1)[0]

                # NPV/PPV
                if len(idx_certain_inact) > 0:
                    validity_inact = np.sum(
                        cp_test_arr_col_str[idx_certain_inact]
                        == labels_fte_col[idx_certain_inact].astype(str)
                    ) / len(cp_test_arr_col_str[idx_certain_inact])

                else:
                    validity_inact = 0

                if len(idx_certain_act) > 0:
                    validity_act = np.sum(
                        cp_test_arr_col_str[idx_certain_act]
                        == labels_fte_col[idx_certain_act].astype(str)
                    ) / len(cp_test_arr_col_str[idx_certain_act])
                else:
                    validity_act = 0

                # validity
                idx_inact_both = np.intersect1d(idx_inact_labels, idx_uncertain_both)
                idx_act_both = np.intersect1d(idx_act_labels, idx_uncertain_both)
                validity_inact = (
                    np.sum(
                        cp_test_arr_col_str[idx_certain_inact]
                        == labels_fte_col[idx_certain_inact].astype(str)
                    )
                    + len(idx_inact_both)
                ) / len(idx_inact_labels)
                validity_act = (
                    np.sum(
                        cp_test_arr_col_str[idx_certain_act]
                        == labels_fte_col[idx_certain_act].astype(str)
                    )
                    + len(idx_act_both)
                ) / len(idx_act_labels)

            else:
                validity_inact = np.nan
                validity_act = np.nan
                validity_inact = np.nan
                validity_act = np.nan
                idx_act_labels = [np.nan]
                idx_inact_labels = [np.nan]

            idxs.append(col)
            unis.append(str(list(uni)))
            e_overall.append(efficiency_overall)
            e_inacts.append(efficiency_inact)
            e_acts.append(efficiency_act)
            e_inacts_nonone.append(efficiency_inact_nonone)
            e_acts_nonone.append(efficiency_act_nonone)
            certain_acts.append(len(idx_certain_act))
            certain_inacts.append(len(idx_certain_inact))
            uncertain_boths.append(len(idx_uncertain_both))
            uncertain_nones.append(len(idx_uncertain_none))
            # labelled
            val_inacts.append(validity_inact)
            val_acts.append(validity_act)
            lit_val_inacts.append(validity_inact)
            lit_val_acts.append(validity_act)
            n_acts.append(len(idx_act_labels))
            n_inacts.append(len(idx_inact_labels))

        except Exception as e:

            idxs.append(col)
            unis.append(np.nan)
            e_overall.append(np.nan)
            e_inacts.append(np.nan)
            e_acts.append(np.nan)
            e_inacts_nonone.append(np.nan)
            e_acts_nonone.append(np.nan)
            certain_acts.append(np.nan)
            certain_inacts.append(np.nan)
            uncertain_boths.append(np.nan)
            uncertain_nones.append(np.nan)
            # labelled
            val_inacts.append(np.nan)
            val_acts.append(np.nan)
            lit_val_inacts.append(np.nan)
            lit_val_acts.append(np.nan)
            n_acts.append(np.nan)
            n_inacts.append(np.nan)

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            print(
                "Error encountered for index {}, during eval_cp function".format(
                    col
                )
            )
            print("Description is below")
            print(e)

    # this will give a hard error if the lists are not the same length
    df_out = pd.DataFrame(
        {
            "index": idxs,
            "cp_values": unis,
            "efficiency_overall": e_overall,
            "efficiency_0": e_inacts,
            "efficiency_1": e_acts,
            "efficiency_nonone_0": e_inacts_nonone,
            "efficiency_nonone_1": e_acts_nonone,
            "n_certain_0": certain_inacts,
            "n_certain_1": certain_acts,
            "n_uncertain_both": uncertain_boths,
            "n_uncertain_none": uncertain_nones
            # labelled
            ,
            "NPV_0": val_inacts,
            "PPV_1": val_acts,
            "validity_0": lit_val_inacts,
            "validity_1": lit_val_acts,
            "n_eval_0": n_inacts,
            "n_eval_1": n_acts,
        }
    )

    return df_out
