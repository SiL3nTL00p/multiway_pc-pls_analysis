"""Utility functions for multi-way PCA and PLS."""

import numpy as np


def unfold(X, mode=0):
    """
    Unfold an R-way array to a 2D matrix along the specified mode.

    Parameters
    ----------
    X : np.ndarray
        R-way array of shape (N, J, K, ...) where mode 0 is the first dimension.
    mode : int
        Mode along which to unfold. Default is 0 (samples).

    Returns
    -------
    X_unfolded : np.ndarray
        2D array of shape (N, J*K*...) if mode=0, or other shapes if mode != 0.
    """
    # Move the specified mode to the front
    X_moved = np.moveaxis(X, mode, 0)
    # Get shape
    shape = X_moved.shape
    # Reshape to 2D
    X_unfolded = X_moved.reshape(shape[0], -1)
    return X_unfolded


def preprocess(X, scale=True, center=True):
    """
    Preprocess array by centering and/or scaling.

    Parameters
    ----------
    X : np.ndarray
        Array of any shape. First dimension is samples (N).
    scale : bool
        If True, scale to unit variance along sample dimension.
    center : bool
        If True, center (subtract mean) along sample dimension.

    Returns
    -------
    X_proc : np.ndarray
        Preprocessed array (same shape as input).
    means : np.ndarray
        Mean values (shape of X[1:]).
    stds : np.ndarray
        Standard deviation values (shape of X[1:]).
    """
    X_proc = X.copy()
    means = None
    stds = None

    if center:
        means = np.mean(X_proc, axis=0, keepdims=True)
        X_proc = X_proc - means
        means = np.squeeze(means, axis=0)

    if scale:
        stds = np.std(X_proc, axis=0, keepdims=True)
        # Avoid division by zero
        stds = np.where(stds == 0, 1.0, stds)
        X_proc = X_proc / stds
        stds = np.squeeze(stds, axis=0)

    return X_proc, means, stds


def reverse_preprocess(X, means=None, stds=None):
    """
    Undo preprocessing (reverse scaling and centering).

    Parameters
    ----------
    X : np.ndarray
        Preprocessed array.
    means : np.ndarray, optional
        Mean values from preprocessing.
    stds : np.ndarray, optional
        Standard deviations from preprocessing.

    Returns
    -------
    X_orig : np.ndarray
        Unpreprocessed array.
    """
    X_orig = X.copy()

    if stds is not None:
        stds = np.expand_dims(stds, axis=0)
        X_orig = X_orig * stds

    if means is not None:
        means = np.expand_dims(means, axis=0)
        X_orig = X_orig + means

    return X_orig


def variance_explained(X, T, P_list):
    """
    Compute percentage of variance explained per component.

    Parameters
    ----------
    X : np.ndarray
        Original R-way array (N, J, K, ...).
    T : np.ndarray
        Score matrix (N, A) where A is number of components.
    P_list : list of np.ndarray
        List of loading arrays. Each P_a has shape (J, K, ...).

    Returns
    -------
    var_explained : np.ndarray
        Percentage variance explained by each component (shape (A,)).
    cumsum_var : np.ndarray
        Cumulative sum of variance explained.
    """
    N = X.shape[0]

    # Total variance in X
    total_var = np.sum(X ** 2)

    # Variance explained by each component
    var_explained_list = []
    for a, P_a in enumerate(P_list):
        # Reconstruct: X_a = t_a ⊗ P_a = outer product
        t_a = T[:, a:a+1]  # shape (N, 1)
        X_recon = np.outer(t_a.ravel(), P_a.ravel()).reshape(X.shape)
        var_a = np.sum(X_recon ** 2)
        var_explained_list.append(var_a)

    var_explained = 100 * np.array(var_explained_list) / total_var
    cumsum_var = np.cumsum(var_explained)

    return var_explained, cumsum_var


def norm_squared(arr):
    """Compute squared Frobenius norm of an array."""
    return np.sum(arr ** 2)


def norm(arr):
    """Compute Frobenius norm of an array."""
    return np.sqrt(norm_squared(arr))
