"""
Numerical example from Wold et al. (1987) - Reproduces Tables 2, 3, 4.

This script reproduces the exact numerical example from the paper,
demonstrating multi-way PCA and PLS on a small 3-way array.
"""

import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, '..')

from multiway_pls import MultiwayPCA, MultiwayPLS, unfold
from sklearn.decomposition import PCA as StandardPCA


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def main():
    # Table 2 data: 3-way X array (3 objects, 2x2 variables)
    X = np.array([
        [[0.424264, 0.565685], [0.565685, 0.424264]],
        [[0.565685, 0.424264], [0.424264, 0.565685]],
        [[0.707101, 0.707101], [0.707101, 0.707101]],
    ])  # shape (3, 2, 2)

    # Response variable
    Y = np.array([[1.0, 1.0], [2.0, 1.5], [3.0, 2.0]])  # shape (3, 2)

    # Test data
    X_test = np.array([[[0.5, 0.6], [0.6, 0.4]]])  # shape (1, 2, 2)

    print_section("MULTIWAY PCA AND PLS ANALYSIS")
    print("Wold et al. (1987) Numerical Example")
    print(f"X shape: {X.shape} (3-way array)")
    print(f"Y shape: {Y.shape} (2-way array)")

    # ====================
    # Part 1: PCA on X
    # ====================
    print_section("PART 1: MULTI-WAY PCA ON X")

    # Multi-way PCA
    pca_mw = MultiwayPCA(n_components=2, center=True, scale=False)
    pca_mw.fit(X)

    print("Multi-way PCA Results:")
    print(f"Number of components: {pca_mw.n_components}")
    print(f"Iterations per component: {pca_mw.n_iter_}")
    print(f"\nExplained variance ratio: {pca_mw.explained_variance_ratio()}")

    # Display scores (Table 3 equivalent)
    print("\nScores T (shape: 3x2):")
    print(pca_mw.T)

    # Display loadings (Table 3 equivalent)
    print("\nLoading Array P_1 (shape: 2x2):")
    print(pca_mw.P_list[0])

    print("\nLoading Array P_2 (shape: 2x2):")
    print(pca_mw.P_list[1])

    # ====================
    # Validation: Compare with standard PCA on unfolded X
    # ====================
    print_section("VALIDATION: COMPARE WITH STANDARD PCA")

    # Unfold X along mode 0 (samples)
    X_unfolded = unfold(X, mode=0)
    print(f"Unfolded X shape: {X_unfolded.shape}")

    # Standard PCA on unfolded X
    pca_std = StandardPCA(n_components=2)
    T_std = pca_std.fit_transform(X_unfolded)

    print("\nStandard PCA Scores:")
    print(T_std)

    print("\nDifference in scores (should be close to zero):")
    print(np.abs(np.abs(pca_mw.T) - np.abs(T_std)))

    # ====================
    # Part 2: PLS on X and Y
    # ====================
    print_section("PART 2: MULTI-WAY PLS ON X AND Y")

    # Multi-way PLS
    pls_mw = MultiwayPLS(n_components=2, center=True, scale=False)
    pls_mw.fit(X, Y)

    print("Multi-way PLS Results:")
    print(f"Number of components: {pls_mw.n_components}")
    print(f"Iterations per component: {pls_mw.n_iter_}")

    # Display scores
    print("\nX-block Scores T (shape: 3x2):")
    print(pls_mw.T)

    print("\nY-block Scores U (shape: 3x2):")
    print(pls_mw.U)

    # Display weights
    print("\nWeight Vector W_1 (shape: 2x2):")
    print(pls_mw.W_list[0])

    print("\nWeight Vector W_2 (shape: 2x2):")
    print(pls_mw.W_list[1])

    # Display loadings
    print("\nLoading Array P_1 (shape: 2x2):")
    print(pls_mw.P_list[0])

    print("\nLoading Array Q_1 (shape: 2):")
    print(pls_mw.Q_list[0])

    # Display inner relation
    print("\nInner Relation Coefficients B (shape: 2):")
    print(pls_mw.B)

    # ====================
    # Part 3: Predictions (Table 4 equivalent)
    # ====================
    print_section("PART 3: PREDICTIONS ON TRAINING SET")

    Y_pred = pls_mw.predict(X)
    print("Predicted Y (training set):")
    print(Y_pred)

    print("\nActual Y:")
    print(Y)

    print("\nPrediction Error (RMSE per response):")
    rmse = np.sqrt(np.mean((Y - Y_pred) ** 2, axis=0))
    print(rmse)

    # ====================
    # Part 4: Test set predictions
    # ====================
    print_section("PART 4: TEST SET PREDICTIONS")

    print(f"Test X shape: {X_test.shape}")
    Y_pred_test = pls_mw.predict(X_test)
    print(f"Predicted Y for test set:")
    print(Y_pred_test)

    # ====================
    # Part 5: Orthogonality checks
    # ====================
    print_section("PART 5: ORTHOGONALITY CHECKS")

    print("T'T (should be diagonal or close to identity if normalized):")
    print(pls_mw.T.T @ pls_mw.T)

    print("\n||P_1||:")
    P1_norm = np.sqrt(np.sum(pls_mw.P_list[0] ** 2))
    print(P1_norm)

    print("\n||P_2||:")
    P2_norm = np.sqrt(np.sum(pls_mw.P_list[1] ** 2))
    print(P2_norm)

    print("\nP_1 * P_2 (should be close to zero if orthogonal):")
    inner_prod = np.sum(pls_mw.P_list[0] * pls_mw.P_list[1])
    print(inner_prod)

    print_section("END OF ANALYSIS")


if __name__ == "__main__":
    main()
