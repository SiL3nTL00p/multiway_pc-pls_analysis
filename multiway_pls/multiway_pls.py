"""Multi-way PLS using NIPALS algorithm (Wold et al., 1987)."""

import numpy as np
from .utils import preprocess, norm_squared


class MultiwayPLS:
    """
    Multi-way Partial Least Squares using NIPALS algorithm.

    Implements the PLS two-block mode A NIPALS algorithm from Wold et al. (1987)
    for R-way X arrays and 2-way or multi-way Y arrays.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS components to extract.
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
        self.T = None  # Scores for X (N, A)
        self.U = None  # Scores for Y (N, A)
        self.W_list = []  # Weight vectors for X
        self.P_list = []  # Loading arrays for X
        self.Q_list = []  # Loading arrays for Y
        self.B = None  # Inner relation coefficients (A,)
        self.means_X = None
        self.stds_X = None
        self.means_Y = None
        self.stds_Y = None
        self.n_iter_ = []

    def fit(self, X, Y):
        """
        Fit the PLS model to data.

        Parameters
        ----------
        X : np.ndarray
            R-way predictor array of shape (N, J, K, ...).
        Y : np.ndarray
            2-way or multi-way response array of shape (N, L, ...).

        Returns
        -------
        self : MultiwayPLS
            Fitted model.
        """
        # Preprocess
        X_proc, self.means_X, self.stds_X = preprocess(X, scale=self.scale, center=self.center)
        Y_proc, self.means_Y, self.stds_Y = preprocess(Y, scale=self.scale, center=self.center)

        N = X_proc.shape[0]
        R_X = X_proc.ndim  # Number of ways in X
        R_Y = Y_proc.ndim  # Number of ways in Y

        # Initialize storage
        self.T = np.zeros((N, self.n_components))
        self.U = np.zeros((N, self.n_components))
        self.W_list = []
        self.P_list = []
        self.Q_list = []
        self.B = np.zeros(self.n_components)
        self.n_iter_ = []

        # Working copies
        E = X_proc.copy()
        F = Y_proc.copy()

        # Extract each component
        for a in range(self.n_components):
            # Step b4: Initialize u with column of F with largest variance
            F_unfolded = F.reshape(N, -1)
            variances = np.var(F_unfolded, axis=0)
            max_idx = np.argmax(variances)
            u = F_unfolded[:, max_idx].copy().astype(np.float64)
            u_norm = np.linalg.norm(u)
            if u_norm > 1e-16:
                u = u / u_norm

            # NIPALS iteration loop (steps b5-b11)
            for iteration in range(self.max_iter):
                t_old = None

                # Step b5: W = u'E (for R-way X)
                # W[j,k,...] = sum_i u[i] * E[i,j,k,...]
                W = np.tensordot(u, E, axes=([0], [0]))

                # Step b6: Normalize W to unit norm
                W_norm = np.sqrt(norm_squared(W))
                if W_norm > 1e-16:
                    W = W / W_norm
                else:
                    break

                # Step b7: t = E..W''
                # t[i] = sum_{j,k,...} E[i,j,k,...] * W[j,k,...]
                t = np.tensordot(E, W, axes=(list(range(1, R_X)), list(range(R_X-1))))
                t_norm = np.linalg.norm(t)
                if t_norm > 1e-16:
                    t = t / t_norm
                else:
                    break

                # Step b8: Convergence check on d
                if t_old is None:
                    t_old = t.copy()
                    d = float('inf')
                else:
                    d = np.sum((t - t_old) ** 2) / (np.sum(t ** 2) + 1e-16)

                if d < self.tol and iteration > 0:
                    break

                # Step b9: Q = t'F (for R_Y-way Y)
                # Q[l,...] = sum_i t[i] * F[i,l,...]
                Q = np.tensordot(t, F, axes=([0], [0]))

                # Step b11: u = F..Q''
                Q_norm_sq = norm_squared(Q)
                if Q_norm_sq > 1e-16:
                    u = np.tensordot(F, Q, axes=(list(range(1, R_Y)), list(range(R_Y-1))))
                    u_norm = np.linalg.norm(u)
                    if u_norm > 1e-16:
                        u = u / u_norm
                    else:
                        break
                else:
                    break

                t_old = t.copy()

            self.n_iter_.append(iteration + 1)

            # Step b12: Final values
            # P = t'E
            P = np.tensordot(t, E, axes=([0], [0]))

            # b_a = t'u (inner relation)
            b_a = np.dot(t, u)

            # Q = t'F
            Q = np.tensordot(t, F, axes=([0], [0]))

            # Store results (t and u are already normalized)
            self.T[:, a] = t
            self.U[:, a] = u
            self.W_list.append(W)
            self.P_list.append(P)
            self.Q_list.append(Q)
            self.B[a] = b_a

            # Step b13: Residuals
            # E = E - t ⊗ P
            t_expanded = t.reshape(-1, *([1] * (R_X - 1)))
            E = E - t_expanded * P

            # F = F - u ⊗ Q
            u_expanded = u.reshape(-1, *([1] * (R_Y - 1)))
            F = F - u_expanded * Q

        return self

    def predict(self, X):
        """
        Predict Y for new objects.

        Parameters
        ----------
        X : np.ndarray
            R-way predictor array of shape (N, J, K, ...).

        Returns
        -------
        Y_pred : np.ndarray
            Predicted response array (same shape as original Y).
        """
        if self.T is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Preprocess X
        N = X.shape[0]
        R_X = X.ndim

        X_proc = X.copy()
        if self.center and self.means_X is not None:
            X_proc = X_proc - np.expand_dims(self.means_X, 0)
        if self.scale and self.stds_X is not None:
            X_proc = X_proc / np.expand_dims(self.stds_X, 0)

        # Compute scores for X using deflated residuals (same as fitting)
        E = X_proc.copy()
        T_new = np.zeros((N, self.n_components))

        for a in range(self.n_components):
            W_a = self.W_list[a]
            # t_a = E..W_a
            t_a = np.tensordot(E, W_a, axes=(list(range(1, R_X)), list(range(R_X-1))))
            t_a_norm = np.linalg.norm(t_a)
            if t_a_norm > 1e-16:
                t_a = t_a / t_a_norm
            T_new[:, a] = t_a

            # Deflate E for next component
            P_a = self.P_list[a]
            t_expanded = t_a.reshape(-1, *([1] * (R_X - 1)))
            E = E - t_expanded * P_a

        # Step b14: Predict Y = sum_a t_a * b_a * Q_a
        # Y_pred[i,l,...] = sum_a T_new[i,a] * B[a] * Q[a,l,...]
        R_Y = len(self.Q_list[0].shape) + 1  # Reconstruct Y dimensionality
        Y_shape = (N,) + self.Q_list[0].shape
        Y_pred = np.zeros(Y_shape)

        for a in range(self.n_components):
            t_a = T_new[:, a]
            b_a = self.B[a]
            Q_a = self.Q_list[a]
            # t_a ⊗ (b_a * Q_a)
            t_expanded = t_a.reshape(-1, *([1] * (R_Y - 1)))
            Y_pred = Y_pred + t_expanded * (b_a * Q_a)

        # Reverse preprocessing of Y
        if self.scale and self.stds_Y is not None:
            Y_pred = Y_pred * np.expand_dims(self.stds_Y, 0)
        if self.center and self.means_Y is not None:
            Y_pred = Y_pred + np.expand_dims(self.means_Y, 0)

        return Y_pred

    def fit_predict(self, X, Y):
        """
        Fit the model and predict Y.

        Parameters
        ----------
        X : np.ndarray
            R-way predictor array.
        Y : np.ndarray
            Response array.

        Returns
        -------
        Y_pred : np.ndarray
            Predicted Y values.
        """
        self.fit(X, Y)
        return self.predict(X)

    def get_loadings_X(self):
        """Get loading arrays for X."""
        return self.P_list

    def get_loadings_Y(self):
        """Get loading arrays for Y."""
        return self.Q_list

    def get_weights_X(self):
        """Get weight vectors for X."""
        return self.W_list
