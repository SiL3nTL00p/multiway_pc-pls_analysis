"""Multi-way PCA using NIPALS algorithm (Wold et al., 1987)."""

import numpy as np
from .utils import preprocess, reverse_preprocess, norm_squared


class MultiwayPCA:
    """
    Multi-way Principal Component Analysis using NIPALS algorithm.

    Implements the generalized PCA NIPALS algorithm from Wold et al. (1987)
    for arrays of arbitrary dimensionality (R-way arrays).

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to extract.
    tol : float, default=1e-10
        Convergence tolerance for NIPALS iterations.
    max_iter : int, default=500
        Maximum number of NIPALS iterations per component.
    scale : bool, default=True
        If True, scale each variable to unit variance.
    center : bool, default=True
        If True, center each variable to zero mean.
    """

    def __init__(self, n_components=2, tol=1e-10, max_iter=500, scale=True, center=True):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.scale = scale
        self.center = center

        # Will be set during fit
        self.T = None  # Scores (N, A)
        self.P_list = []  # List of loading arrays
        self.means = None
        self.stds = None
        self.eigenvalues = []
        self.n_iter_ = []  # Iterations per component

    def fit(self, X):
        """
        Fit the PCA model to data.

        Parameters
        ----------
        X : np.ndarray
            R-way array of shape (N, J, K, ...).

        Returns
        -------
        self : MultiwayPCA
            Fitted model.
        """
        # Preprocess
        X_proc, self.means, self.stds = preprocess(X, scale=self.scale, center=self.center)

        N = X_proc.shape[0]
        R = X_proc.ndim  # Number of ways

        # Initialize storage
        self.T = np.zeros((N, self.n_components))
        self.P_list = []
        self.eigenvalues = []
        self.n_iter_ = []

        # Working copy
        E = X_proc.copy()

        # Extract each component
        for a in range(self.n_components):
            # Step a4: Initialize t with the column of E with largest variance
            # Unfold E along first mode (samples)
            E_unfolded = E.reshape(N, -1)
            variances = np.var(E_unfolded, axis=0)
            max_idx = np.argmax(variances)
            t = E_unfolded[:, max_idx].copy().astype(np.float64)

            # Normalize to unit length
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-16:
                t = t / t_norm

            # NIPALS iteration loop (steps a5-a8)
            for iteration in range(self.max_iter):
                t_old = t.copy()

                # Step a5: P = t'X (generalized inner product)
                # P[j,k,...] = sum_i t[i] * E[i,j,k,...]
                P = np.tensordot(t, E, axes=([0], [0]))

                # Step a7: t = E..P'' / ||P||^2 (generalized outer product)
                # t[i] = sum_{j,k,...} E[i,j,k,...] * P[j,k,...]
                P_norm_sq = norm_squared(P)
                if P_norm_sq > 1e-16:
                    t_new = np.tensordot(E, P, axes=(list(range(1, R)), list(range(R-1))))
                    t_new_norm = np.linalg.norm(t_new)
                    if t_new_norm > 1e-16:
                        t = t_new / t_new_norm
                    else:
                        break
                else:
                    break

                # Step a8: Convergence check
                t_norm = np.linalg.norm(t)
                if t_norm > 1e-16:
                    d = np.sum((t - t_old) ** 2) / (np.sum(t ** 2) + 1e-16)
                else:
                    d = 0

                if d < self.tol:
                    break

            self.n_iter_.append(iteration + 1)

            # Store t (normalized)
            self.T[:, a] = t

            # Compute final P
            P = np.tensordot(t, E, axes=([0], [0]))
            self.P_list.append(P)

            # Compute eigenvalue (variance explained by this component)
            eigenvalue = norm_squared(P)
            self.eigenvalues.append(eigenvalue)

            # Step a9: Deflate (compute residuals)
            # E = E - t ⊗ P
            # Compute outer product: reshape for broadcasting
            t_expanded = t.reshape(-1, *([1] * (R - 1)))
            E = E - t_expanded * P

        return self

    def transform(self, X):
        """
        Transform data using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            R-way array of shape (N, J, K, ...).

        Returns
        -------
        T : np.ndarray
            Score matrix (N, n_components).
        """
        if self.T is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Apply preprocessing using stored parameters
        X_proc = X.copy()
        if self.center and self.means is not None:
            X_proc = X_proc - np.expand_dims(self.means, 0)
        if self.scale and self.stds is not None:
            X_proc = X_proc / np.expand_dims(self.stds, 0)

        N = X_proc.shape[0]
        R = X_proc.ndim
        T = np.zeros((N, self.n_components))

        # Compute scores using stored loading arrays
        for a in range(self.n_components):
            P_a = self.P_list[a]
            # t_a = X..P_a'' (no normalization, since t is stored normalized)
            t_a = np.tensordot(X_proc, P_a, axes=(list(range(1, R)), list(range(R-1))))
            t_a_norm = np.linalg.norm(t_a)
            if t_a_norm > 1e-16:
                t_a = t_a / t_a_norm
            T[:, a] = t_a

        return T

    def fit_transform(self, X):
        """
        Fit the model and transform data.

        Parameters
        ----------
        X : np.ndarray
            R-way array of shape (N, J, K, ...).

        Returns
        -------
        T : np.ndarray
            Score matrix (N, n_components).
        """
        self.fit(X)
        return self.T

    def get_loadings(self):
        """
        Get the loading arrays for each component.

        Returns
        -------
        P_list : list of np.ndarray
            Loading arrays.
        """
        return self.P_list

    def explained_variance_ratio(self):
        """
        Compute the proportion of variance explained by each component.

        Returns
        -------
        ratio : np.ndarray
            Variance explained ratio for each component.
        """
        eigenvalues = np.array(self.eigenvalues)
        return eigenvalues / np.sum(eigenvalues)
