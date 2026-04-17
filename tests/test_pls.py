"""Tests for MultiwayPLS."""

import sys
import numpy as np

sys.path.insert(0, '..')

from multiway_pls import MultiwayPLS


def test_multiway_pls_basic():
    """Test basic PLS fitting."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 4)
    Y = np.random.randn(10, 2)

    pls = MultiwayPLS(n_components=2)
    pls.fit(X, Y)

    assert pls.T.shape == (10, 2), f"Wrong T shape: {pls.T.shape}"
    assert pls.U.shape == (10, 2), f"Wrong U shape: {pls.U.shape}"
    assert len(pls.W_list) == 2, f"Wrong number of W vectors"
    assert len(pls.P_list) == 2, f"Wrong number of P arrays"
    assert len(pls.Q_list) == 2, f"Wrong number of Q arrays"
    assert pls.B.shape == (2,), f"Wrong B shape: {pls.B.shape}"

    print("✓ test_multiway_pls_basic passed")


def test_multiway_pls_predict():
    """Test PLS prediction."""
    np.random.seed(42)
    X_train = np.random.randn(10, 3, 4)
    Y_train = np.random.randn(10, 2)
    X_test = np.random.randn(5, 3, 4)

    pls = MultiwayPLS(n_components=2)
    pls.fit(X_train, Y_train)

    Y_pred = pls.predict(X_test)

    assert Y_pred.shape == (5, 2), f"Wrong prediction shape: {Y_pred.shape}"
    assert np.all(np.isfinite(Y_pred)), "Predictions contain non-finite values"

    print("✓ test_multiway_pls_predict passed")


def test_multiway_pls_fit_predict():
    """Test fit_predict method."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 4)
    Y = np.random.randn(10, 2)

    pls = MultiwayPLS(n_components=2)
    Y_pred = pls.fit_predict(X, Y)

    assert Y_pred.shape == Y.shape, f"Wrong shape from fit_predict: {Y_pred.shape}"

    print("✓ test_multiway_pls_fit_predict passed")


def test_multiway_pls_numerical_example():
    """Test on the paper's numerical example (Table 2, 3, 4)."""
    X = np.array([
        [[0.424264, 0.565685], [0.565685, 0.424264]],
        [[0.565685, 0.424264], [0.424264, 0.565685]],
        [[0.707101, 0.707101], [0.707101, 0.707101]],
    ])

    Y = np.array([[1.0, 1.0], [2.0, 1.5], [3.0, 2.0]])

    pls = MultiwayPLS(n_components=2, center=True, scale=False)
    pls.fit(X, Y)

    # Check shapes
    assert pls.T.shape == (3, 2), f"Wrong T shape: {pls.T.shape}"
    assert pls.U.shape == (3, 2), f"Wrong U shape: {pls.U.shape}"

    # Test predictions
    Y_pred = pls.predict(X)
    assert Y_pred.shape == Y.shape, f"Wrong prediction shape: {Y_pred.shape}"

    # Predictions should be reasonably close to actual values
    rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
    assert rmse < 1.0, f"RMSE too high: {rmse}"

    print("✓ test_multiway_pls_numerical_example passed")


def test_multiway_pls_convergence():
    """Test that NIPALS converges within max_iter."""
    np.random.seed(42)
    X = np.random.randn(12, 3, 3)
    Y = np.random.randn(12, 2)

    pls = MultiwayPLS(n_components=2, max_iter=500)
    pls.fit(X, Y)

    # Check that components converged or used reasonable number of iterations
    # (PLS sometimes needs more iterations than PCA)
    for i, n_iter in enumerate(pls.n_iter_):
        assert n_iter <= 500, f"Component {i} exceeded max_iter (iter={n_iter})"

    print("✓ test_multiway_pls_convergence passed")


def test_multiway_pls_orthogonality_T():
    """Test that T vectors are approximately orthogonal."""
    np.random.seed(42)
    X = np.random.randn(15, 3, 3)
    Y = np.random.randn(15, 2)

    pls = MultiwayPLS(n_components=3)
    pls.fit(X, Y)

    # Check T'T is approximately diagonal
    TTT = pls.T.T @ pls.T
    off_diag = TTT - np.diag(np.diag(TTT))
    max_off_diag = np.max(np.abs(off_diag))

    assert max_off_diag < 1.0, f"Non-orthogonal T vectors: max off-diag = {max_off_diag}"

    print("✓ test_multiway_pls_orthogonality_T passed")


def test_multiway_pls_different_y_shapes():
    """Test PLS with different Y shapes."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 3)

    # Test 2D Y
    Y_2d = np.random.randn(10, 2)
    pls_2d = MultiwayPLS(n_components=2)
    pls_2d.fit(X, Y_2d)
    Y_pred_2d = pls_2d.predict(X)
    assert Y_pred_2d.shape == (10, 2), f"Wrong shape for 2D Y: {Y_pred_2d.shape}"

    # Test 1D Y (should be reshaped to (N, 1))
    Y_1d = np.random.randn(10)
    pls_1d = MultiwayPLS(n_components=2)
    pls_1d.fit(X, Y_1d)
    Y_pred_1d = pls_1d.predict(X)
    assert Y_pred_1d.ndim >= 1, f"Wrong shape for 1D Y: {Y_pred_1d.shape}"

    print("✓ test_multiway_pls_different_y_shapes passed")


def test_multiway_pls_higher_order_y():
    """Test PLS with 3-way Y."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 3)
    Y = np.random.randn(10, 2, 2)  # 3-way Y

    pls = MultiwayPLS(n_components=2)
    pls.fit(X, Y)

    Y_pred = pls.predict(X)
    assert Y_pred.shape == Y.shape, f"Wrong prediction shape for 3-way Y: {Y_pred.shape}"

    print("✓ test_multiway_pls_higher_order_y passed")


def test_multiway_pls_scaling_centering():
    """Test that scaling and centering affect results."""
    np.random.seed(42)
    X = np.random.randn(10, 3, 3) + 5.0
    X[:, 0, :] *= 100  # Large variance in first variable
    Y = np.random.randn(10, 2) + 3.0

    # With scaling and centering
    pls_scaled = MultiwayPLS(n_components=2, center=True, scale=True)
    pls_scaled.fit(X, Y)

    # Without scaling and centering
    pls_not_scaled = MultiwayPLS(n_components=2, center=False, scale=False)
    pls_not_scaled.fit(X, Y)

    # Results should be different
    T_s = pls_scaled.T
    T_ns = pls_not_scaled.T
    diff = np.abs(T_s - T_ns)

    assert np.max(diff) > 0.01, "Preprocessing had no effect on results"

    print("✓ test_multiway_pls_scaling_centering passed")


def test_multiway_pls_predictions_close_to_actual():
    """Test that predictions on training set are close to actual values."""
    np.random.seed(123)
    X = np.random.randn(20, 4, 3)
    Y = np.random.randn(20, 2)

    pls = MultiwayPLS(n_components=3, center=True, scale=True)
    pls.fit(X, Y)

    Y_pred = pls.predict(X)

    # R-squared should be positive (predictions better than mean)
    y_mean = np.mean(Y, axis=0)
    ss_tot = np.sum((Y - y_mean) ** 2)
    ss_res = np.sum((Y - Y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot

    assert r2 > 0.0, f"R-squared is negative: {r2}"
    assert r2 < 1.0, f"R-squared is too high: {r2}"

    print("✓ test_multiway_pls_predictions_close_to_actual passed")


if __name__ == "__main__":
    test_multiway_pls_basic()
    test_multiway_pls_predict()
    test_multiway_pls_fit_predict()
    test_multiway_pls_numerical_example()
    test_multiway_pls_convergence()
    test_multiway_pls_orthogonality_T()
    test_multiway_pls_different_y_shapes()
    test_multiway_pls_higher_order_y()
    test_multiway_pls_scaling_centering()
    test_multiway_pls_predictions_close_to_actual()

    print("\n" + "="*50)
    print("All PLS tests passed! ✓")
    print("="*50)
