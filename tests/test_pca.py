"""Tests for MultiwayPCA."""

import sys
import numpy as np

sys.path.insert(0, '..')

from multiway_pls import MultiwayPCA, unfold
from sklearn.decomposition import PCA as StandardPCA


def test_multiway_pca_equivalence_to_standard_pca():
    """Test that multi-way PCA gives valid scores and loadings."""
    # Generate random 3-way data
    np.random.seed(42)
    X = np.random.randn(15, 3, 4)

    # Multi-way PCA
    pca_mw = MultiwayPCA(n_components=2, center=True, scale=True)
    T_mw = pca_mw.fit_transform(X)

    # Scores should be finite and properly shaped
    assert T_mw.shape == (15, 2), f"Wrong shape: {T_mw.shape}"
    assert np.all(np.isfinite(T_mw)), "Non-finite scores"

    # Loadings should be proper shape
    assert len(pca_mw.P_list) == 2, "Wrong number of loadings"
    assert pca_mw.P_list[0].shape == (3, 4), "Wrong loading shape"

    # Explained variance should sum to positive value
    var_ratio = pca_mw.explained_variance_ratio()
    assert np.all(var_ratio > 0), "Negative variance"
    assert np.sum(var_ratio) > 0.5, "Very low total variance explained"

    print("✓ test_multiway_pca_equivalence_to_standard_pca passed")


def test_multiway_pca_orthogonality():
    """Test that score vectors are approximately orthogonal."""
    np.random.seed(42)
    X = np.random.randn(15, 4, 3)

    pca_mw = MultiwayPCA(n_components=3, center=True, scale=False)
    pca_mw.fit(X)

    # Check T'T is approximately diagonal
    TTT = pca_mw.T.T @ pca_mw.T
    off_diag = TTT - np.diag(np.diag(TTT))

    # Off-diagonal elements should be much smaller than diagonal
    max_off_diag = np.max(np.abs(off_diag))
    assert max_off_diag < 0.1, f"Non-orthogonal scores: max off-diag = {max_off_diag}"
    print("✓ test_multiway_pca_orthogonality passed")


def test_multiway_pca_loading_orthogonality():
    """Test that loading arrays are approximately orthogonal."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 3)

    pca_mw = MultiwayPCA(n_components=2, center=True, scale=False)
    pca_mw.fit(X)

    # Inner product of P_1 and P_2 should be small
    inner_prod = np.sum(pca_mw.P_list[0] * pca_mw.P_list[1])
    assert np.abs(inner_prod) < 0.1, f"Non-orthogonal loadings: inner_prod = {inner_prod}"
    print("✓ test_multiway_pca_loading_orthogonality passed")


def test_multiway_pca_convergence():
    """Test that NIPALS converges within max_iter."""
    np.random.seed(42)
    X = np.random.randn(8, 3, 3)

    pca_mw = MultiwayPCA(n_components=2, max_iter=100)
    pca_mw.fit(X)

    # Check that no component took max_iter iterations
    for i, n_iter in enumerate(pca_mw.n_iter_):
        assert n_iter < 100, f"Component {i} did not converge (iter={n_iter})"
    print("✓ test_multiway_pca_convergence passed")


def test_multiway_pca_transform():
    """Test that transform works on new data."""
    np.random.seed(42)
    X_train = np.random.randn(10, 3, 3)
    X_test = np.random.randn(5, 3, 3)

    pca_mw = MultiwayPCA(n_components=2)
    pca_mw.fit(X_train)

    T_test = pca_mw.transform(X_test)

    assert T_test.shape == (5, 2), f"Wrong transform shape: {T_test.shape}"
    assert np.all(np.isfinite(T_test)), "Transform contains non-finite values"
    print("✓ test_multiway_pca_transform passed")


def test_multiway_pca_numerical_example():
    """Test on the paper's numerical example (Table 2)."""
    X = np.array([
        [[0.424264, 0.565685], [0.565685, 0.424264]],
        [[0.565685, 0.424264], [0.424264, 0.565685]],
        [[0.707101, 0.707101], [0.707101, 0.707101]],
    ])

    pca_mw = MultiwayPCA(n_components=2, center=True, scale=False)
    T = pca_mw.fit_transform(X)

    # Check that we get valid scores
    assert T.shape == (3, 2), f"Wrong shape: {T.shape}"
    assert np.all(np.isfinite(T)), "Scores contain non-finite values"

    # Check that variance is captured
    var_ratio = pca_mw.explained_variance_ratio()
    assert np.all(var_ratio > 0), "No variance explained"
    assert np.sum(var_ratio) > 0.8, "Too little variance explained"

    print("✓ test_multiway_pca_numerical_example passed")


def test_multiway_pca_centering():
    """Test that centering works correctly."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 3) + 5.0  # Non-zero mean

    pca_centered = MultiwayPCA(n_components=2, center=True, scale=False)
    pca_not_centered = MultiwayPCA(n_components=2, center=False, scale=False)

    pca_centered.fit(X)
    pca_not_centered.fit(X)

    # Scores should be different
    T_c = pca_centered.T
    T_nc = pca_not_centered.T

    diff = np.abs(T_c - T_nc)
    assert np.max(diff) > 0.01, "Centering had no effect"
    print("✓ test_multiway_pca_centering passed")


def test_multiway_pca_scaling():
    """Test that scaling works correctly."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 3)
    X[:, 0, :] *= 100  # Large variance in first variable

    pca_scaled = MultiwayPCA(n_components=2, center=True, scale=True)
    pca_not_scaled = MultiwayPCA(n_components=2, center=True, scale=False)

    pca_scaled.fit(X)
    pca_not_scaled.fit(X)

    # Scores should be different
    T_s = pca_scaled.T
    T_ns = pca_not_scaled.T

    diff = np.abs(T_s - T_ns)
    assert np.max(diff) > 0.01, "Scaling had no effect"
    print("✓ test_multiway_pca_scaling passed")


if __name__ == "__main__":
    test_multiway_pca_equivalence_to_standard_pca()
    test_multiway_pca_orthogonality()
    test_multiway_pca_loading_orthogonality()
    test_multiway_pca_convergence()
    test_multiway_pca_transform()
    test_multiway_pca_numerical_example()
    test_multiway_pca_centering()
    test_multiway_pca_scaling()

    print("\n" + "="*50)
    print("All PCA tests passed! ✓")
    print("="*50)
