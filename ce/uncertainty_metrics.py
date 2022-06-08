from scipy.stats import entropy
import numpy as np


def calcIG(arr: np.ndarray):
    """
    Information gain calculation
    Expects 3D np array with dimensions:
        1st : compounds
        2nd : tasks (predictions of 1 out of 2 classes for binary classification)
        3rd : repeats
    """
    assert len(arr.shape) == 3
    first = entropy(
        np.dstack([arr.mean(axis=2), 1.0 - arr.mean(axis=2)]), axis=2, base=2
    )
    second = np.mean(
        entropy(np.stack([arr, 1.0 - arr], axis=-1), axis=-1, base=2), axis=2
    )
    delta = first - second
    return delta


def calcVar(preds: np.ndarray):
    """
    Calculating variance along the repeats
    Expects 3D np array with dimensions:
        1st : compounds
        2nd : tasks (predictions of 1 out of 2 classes for binary classification)
        3rd : repeats
    """
    assert len(preds.shape) == 3
    return np.var(preds, (2))


def calcKL(pre_arr=None, post_arr=None):
    """
    KL divergence
    Expects 3D np array with dimensions:
        1st : compounds
        2nd : tasks (predictions of 1 out of 2 classes for binary classification)
        3rd : repeats
    """
    assert len(pre_arr.shape) == 3
    assert len(post_arr.shape) == 3
    pre_arr = np.stack([pre_arr, 1.0 - pre_arr], axis=-1)
    post_arr = np.stack([post_arr, 1.0 - post_arr], axis=-1)
    kl = entropy(
        post_arr, qk=pre_arr, axis=2, base=2
    )  # according to option doc : qk = pre
    kl = np.sum(kl, axis=-1)
    return kl
