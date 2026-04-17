"""
HPLC-UV Example from Wold et al. (1987) - Reproduces Tables 5, 6.

This script generates synthetic HPLC-UV data matching the paper description
and demonstrates multi-way PLS analysis.
"""

import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, '..')

from multiway_pls import MultiwayPLS


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def generate_hplc_data(seed=42):
    """
    Generate synthetic HPLC-UV data for mixtures of anthracene and phenanthrene.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_calib : np.ndarray
        Shape (6, 10, 10) - calibration samples × wavelengths × time
    Y_calib : np.ndarray
        Shape (6, 2) - concentrations of anthracene and phenanthrene
    X_test : np.ndarray
        Shape (4, 10, 10) - test samples
    Y_test : np.ndarray
        Shape (4, 2) - true test concentrations
    """
    np.random.seed(seed)

    # Design of experiments for calibration set
    # 6 samples with varying concentrations
    anthracene_conc = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    phenanthrene_conc = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    Y_calib = np.column_stack([anthracene_conc, phenanthrene_conc])

    # Generate 3-way X data: 6 samples × 10 wavelengths × 10 time points
    # Synthetic spectral signatures that depend on concentrations
    X_calib = np.zeros((6, 10, 10))

    for i in range(6):
        # Base spectrum for anthracene (stronger at higher wavelengths)
        anth_spectrum = anthracene_conc[i] * np.outer(
            np.linspace(0.5, 1.0, 10),  # Wavelength response
            np.exp(-((np.arange(10) - 5) ** 2) / 5)  # Time profile (Gaussian peak)
        )

        # Base spectrum for phenanthrene (peak at different wavelength)
        phen_spectrum = phenanthrene_conc[i] * np.outer(
            np.linspace(0.3, 0.8, 10),  # Different wavelength response
            np.exp(-((np.arange(10) - 6) ** 2) / 4)  # Peak at different time
        )

        # Combine spectra with some noise
        X_calib[i] = anth_spectrum + phen_spectrum + 0.01 * np.random.randn(10, 10)

    # Test set: 4 new samples
    anthracene_test = np.array([1.3, 2.2, 2.8, 3.2])
    phenanthrene_test = np.array([1.2, 1.8, 2.3, 2.9])

    Y_test = np.column_stack([anthracene_test, phenanthrene_test])

    # Generate test X data
    X_test = np.zeros((4, 10, 10))

    for i in range(4):
        anth_spectrum = anthracene_test[i] * np.outer(
            np.linspace(0.5, 1.0, 10),
            np.exp(-((np.arange(10) - 5) ** 2) / 5)
        )
        phen_spectrum = phenanthrene_test[i] * np.outer(
            np.linspace(0.3, 0.8, 10),
            np.exp(-((np.arange(10) - 6) ** 2) / 4)
        )
        X_test[i] = anth_spectrum + phen_spectrum + 0.01 * np.random.randn(10, 10)

    return X_calib, Y_calib, X_test, Y_test


def main():
    print_section("HPLC-UV MULTIWAY PLS ANALYSIS")
    print("Wold et al. (1987) Example Structure")

    # Generate data
    X_calib, Y_calib, X_test, Y_test = generate_hplc_data()

    print(f"Calibration set:")
    print(f"  X shape: {X_calib.shape} (samples × wavelengths × time)")
    print(f"  Y shape: {Y_calib.shape} (samples × constituents)")
    print(f"\nTest set:")
    print(f"  X shape: {X_test.shape}")
    print(f"  Y shape: {Y_test.shape}")

    print("\nCalibration Y (Anthracene, Phenanthrene):")
    print(Y_calib)

    print("\nTest Y (True values):")
    print(Y_test)

    # ====================
    # Part 1: Unconstrained PLS
    # ====================
    print_section("UNCONSTRAINED PLS ANALYSIS")

    # Fit unconstrained PLS
    pls_unc = MultiwayPLS(n_components=3, center=True, scale=True)
    pls_unc.fit(X_calib, Y_calib)

    print(f"Number of components: {pls_unc.n_components}")
    print(f"Iterations per component: {pls_unc.n_iter_}")

    # Table 5 equivalent: Variance explained per component
    print("\nVariance Explained per Component:")
    print("\nX-block:")
    for a in range(pls_unc.n_components):
        # Compute variance explained
        t_a = pls_unc.T[:, a]
        X_recon = np.outer(t_a, pls_unc.P_list[a].ravel()).reshape(X_calib.shape)
        var_a = np.sum(X_recon ** 2) / np.sum(X_calib ** 2)
        print(f"  Component {a+1}: {var_a*100:.2f}%")

    print("\nY-block:")
    for a in range(pls_unc.n_components):
        # Compute Y variance explained
        u_a = pls_unc.U[:, a]
        Y_recon = np.outer(u_a, pls_unc.Q_list[a].ravel()).reshape(Y_calib.shape)
        var_a = np.sum(Y_recon ** 2) / np.sum(Y_calib ** 2)
        print(f"  Component {a+1}: {var_a*100:.2f}%")

    # Calibration predictions (Table 5 structure)
    Y_pred_calib = pls_unc.predict(X_calib)

    print("\nCalibration Predictions:")
    print("Predicted Y:")
    print(Y_pred_calib)

    print("\nActual Y:")
    print(Y_calib)

    print("\nCalibration RMSE (per response variable):")
    rmse_calib = np.sqrt(np.mean((Y_calib - Y_pred_calib) ** 2, axis=0))
    print(f"  Anthracene: {rmse_calib[0]:.4f}")
    print(f"  Phenanthrene: {rmse_calib[1]:.4f}")

    # ====================
    # Part 2: Test set predictions (Table 6 equivalent)
    # ====================
    print_section("TEST SET PREDICTIONS")

    Y_pred_test = pls_unc.predict(X_test)

    print("Predicted concentrations (Test set):")
    print("(Anthracene, Phenanthrene)")
    for i, (pred, true) in enumerate(zip(Y_pred_test, Y_test)):
        print(f"Sample {i+1}:")
        print(f"  Predicted: Anthracene={pred[0]:.3f}, Phenanthrene={pred[1]:.3f}")
        print(f"  True:      Anthracene={true[0]:.3f}, Phenanthrene={true[1]:.3f}")
        print(f"  Error:     Anthracene={abs(pred[0]-true[0]):.3f}, "
              f"Phenanthrene={abs(pred[1]-true[1]):.3f}")

    print("\nTest Set RMSE (per response variable):")
    rmse_test = np.sqrt(np.mean((Y_test - Y_pred_test) ** 2, axis=0))
    print(f"  Anthracene: {rmse_test[0]:.4f}")
    print(f"  Phenanthrene: {rmse_test[1]:.4f}")

    # ====================
    # Part 3: Variance comparison across components
    # ====================
    print_section("CUMULATIVE VARIANCE EXPLAINED")

    var_X = np.zeros(pls_unc.n_components)
    var_Y = np.zeros(pls_unc.n_components)

    for a in range(pls_unc.n_components):
        # X variance
        t_a = pls_unc.T[:, a]
        X_recon = np.outer(t_a, pls_unc.P_list[a].ravel()).reshape(X_calib.shape)
        var_X[a] = np.sum(X_recon ** 2) / np.sum(X_calib ** 2) * 100

        # Y variance
        u_a = pls_unc.U[:, a]
        Y_recon = np.outer(u_a, pls_unc.Q_list[a].ravel()).reshape(Y_calib.shape)
        var_Y[a] = np.sum(Y_recon ** 2) / np.sum(Y_calib ** 2) * 100

    print("X-block variance:")
    for a in range(pls_unc.n_components):
        cumsum = np.sum(var_X[:a+1])
        print(f"  Component {a+1}: {var_X[a]:.2f}% (Cumulative: {cumsum:.2f}%)")

    print("\nY-block variance:")
    for a in range(pls_unc.n_components):
        cumsum = np.sum(var_Y[:a+1])
        print(f"  Component {a+1}: {var_Y[a]:.2f}% (Cumulative: {cumsum:.2f}%)")

    print_section("END OF HPLC ANALYSIS")


if __name__ == "__main__":
    main()
