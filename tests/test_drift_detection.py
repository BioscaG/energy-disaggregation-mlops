"""
Tests for data drift detection and model robustness (M27).

This test suite validates:
1. Data drift detection capabilities (covariate shift, concept drift)
2. Model robustness under various perturbations (noise, scale, missing data)
3. Statistical measures for drift quantification
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from energy_dissagregation_mlops.data import MyDataset
from energy_dissagregation_mlops.drift_detection import (
    DataDriftDetector,
    RobustnessAnalyzer,
    visualize_drift_analysis,
    visualize_robustness_analysis,
)
from energy_dissagregation_mlops.model import Model


@pytest.fixture
def reference_data():
    """Generate reference (training) distribution."""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=20, size=(1000,))


@pytest.fixture
def drifted_data():
    """Generate drifted (test) distribution with mean shift."""
    np.random.seed(43)
    return np.random.normal(loc=130, scale=20, size=(1000,))


@pytest.fixture
def model_device():
    """Get device for model."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def test_model(model_device):
    """Create and prepare test model."""
    model = Model(window_size=1024)
    model = model.to(model_device)
    model.eval()
    return model


@pytest.fixture
def sample_data(model_device):
    """Generate sample input-output data for testing."""
    x = torch.randn(32, 1, 1024, device=model_device)  # [B, 1, T]
    y = torch.randn(32, 1, 1024, device=model_device)  # [B, 1, T]
    return x, y


class TestDataDriftDetection:
    """Test data drift detection methods."""

    def test_ks_test_detects_shift(self, reference_data, drifted_data):
        """Test that KS test detects distribution shift."""
        result = DataDriftDetector.kolmogorov_smirnov_test(reference_data, drifted_data)

        assert "statistic" in result
        assert "p_value" in result
        assert "drift_detected" in result
        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05
        logger.info(f"KS test detected drift: statistic={result['statistic']:.4f}")

    def test_ks_test_no_shift(self, reference_data):
        """Test that KS test doesn't detect shift in identical distributions."""
        result = DataDriftDetector.kolmogorov_smirnov_test(reference_data, reference_data.copy())

        assert "drift_detected" in result
        assert result["drift_detected"] is False
        assert result["p_value"] > 0.05

    def test_psi_calculation(self, reference_data, drifted_data):
        """Test PSI calculation."""
        psi = DataDriftDetector.population_stability_index(reference_data, drifted_data)

        assert isinstance(psi, float)
        assert psi > 0
        # Drifted data should have significant PSI
        assert psi > 0.1

    def test_psi_stable_distribution(self, reference_data):
        """Test PSI for stable (non-drifted) distribution."""
        psi = DataDriftDetector.population_stability_index(reference_data, reference_data.copy())

        assert psi < 0.05  # Should be very small

    def test_mmd_calculation(self, reference_data, drifted_data):
        """Test Maximum Mean Discrepancy calculation."""
        mmd = DataDriftDetector.maximum_mean_discrepancy(reference_data, drifted_data)

        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_mmd_rbf_kernel(self, reference_data, drifted_data):
        """Test MMD with RBF kernel."""
        mmd_rbf = DataDriftDetector.maximum_mean_discrepancy(reference_data, drifted_data, kernel="rbf")
        assert mmd_rbf >= 0

    def test_mmd_linear_kernel(self, reference_data, drifted_data):
        """Test MMD with linear kernel."""
        mmd_linear = DataDriftDetector.maximum_mean_discrepancy(reference_data, drifted_data, kernel="linear")
        assert mmd_linear >= 0

    def test_compare_distributions(self, reference_data, drifted_data):
        """Test comprehensive distribution comparison."""
        result = DataDriftDetector.compare_distributions(reference_data, drifted_data, feature_name="test_feature")

        assert result["feature"] == "test_feature"
        assert "reference_mean" in result
        assert "current_mean" in result
        assert "mean_shift" in result
        assert "ks_test" in result
        assert "psi" in result
        assert "mmd" in result

        # Check that mean shift is detected
        assert result["mean_shift"] > 0
        logger.info(f"Mean shift detected: {result['mean_shift']:.4f}")

    def test_drift_detection_with_real_data(self):
        """Test drift detection with simulated energy data."""
        np.random.seed(42)

        # Reference: stable baseline power
        reference_mains = np.random.normal(loc=500, scale=50, size=(1000,))

        # Drifted: shifted baseline (sensor degradation)
        drifted_mains = np.random.normal(loc=550, scale=50, size=(1000,))

        result = DataDriftDetector.compare_distributions(reference_mains, drifted_mains, "mains_power")

        assert result["ks_test"]["drift_detected"]
        assert result["psi"] > 0.1
        logger.info(f"Energy data drift detected: PSI={result['psi']:.4f}")


class TestRobustnessAnalyzer:
    """Test model robustness against perturbations."""

    def test_analyzer_initialization(self, test_model, model_device):
        """Test RobustnessAnalyzer initialization."""
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        assert analyzer.model is test_model
        assert analyzer.device == model_device

    def test_gaussian_noise_perturbation(self, sample_data, test_model, model_device):
        """Test Gaussian noise perturbation generation."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        noise_levels = [0.01, 0.05, 0.1]
        noisy_data = analyzer.add_gaussian_noise(x.cpu().numpy(), noise_levels)

        assert len(noisy_data) == len(noise_levels)
        for level in noise_levels:
            assert noisy_data[level].shape == x.shape
            # Check that noise was actually added
            assert not np.allclose(noisy_data[level], x.cpu().numpy())

    def test_amplitude_scaling_perturbation(self, sample_data, test_model, model_device):
        """Test amplitude scaling perturbation."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        scale_factors = [0.5, 1.0, 1.5]
        scaled_data = analyzer.scale_amplitude(x.cpu().numpy(), scale_factors)

        assert len(scaled_data) == len(scale_factors)
        assert np.allclose(scaled_data[1.0], x.cpu().numpy())
        assert np.allclose(scaled_data[0.5], x.cpu().numpy() * 0.5)

    def test_missing_data_perturbation(self, sample_data, test_model, model_device):
        """Test missing data simulation."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        missing_rates = [0.0, 0.1, 0.2]
        missing_data = analyzer.add_missing_data(x.cpu().numpy(), missing_rates)

        assert len(missing_data) == len(missing_rates)
        assert np.allclose(missing_data[0.0], x.cpu().numpy())  # No missing data

        # Check that data was zeroed out
        for rate in missing_rates[1:]:
            num_zeros = np.sum(missing_data[rate] == 0)
            assert num_zeros > 0

    def test_robustness_to_noise(self, sample_data, test_model, model_device):
        """Test model robustness to Gaussian noise."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        results = analyzer.evaluate_robustness(
            x, y, perturbation_type="noise", perturbation_levels=[0.0, 0.01, 0.05, 0.1]
        )

        assert results["perturbation_type"] == "noise"
        assert "baseline_metric" in results
        assert "metrics_by_level" in results
        assert len(results["metrics_by_level"]) == 4

        # Check that all metrics are computed
        for level, metrics in results["metrics_by_level"].items():
            assert "mse" in metrics
            assert "rmse" in metrics
            assert "mae" in metrics
            assert all(m > 0 for m in [metrics["mse"], metrics["rmse"], metrics["mae"]])

    def test_robustness_to_scale(self, sample_data, test_model, model_device):
        """Test model robustness to amplitude scaling."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        results = analyzer.evaluate_robustness(
            x, y, perturbation_type="scale", perturbation_levels=[0.5, 0.8, 1.0, 1.2, 1.5]
        )

        assert results["perturbation_type"] == "scale"
        assert len(results["metrics_by_level"]) == 5

    def test_robustness_to_missing_data(self, sample_data, test_model, model_device):
        """Test model robustness to missing data."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        results = analyzer.evaluate_robustness(
            x, y, perturbation_type="missing", perturbation_levels=[0.0, 0.05, 0.1, 0.2]
        )

        assert results["perturbation_type"] == "missing"
        assert len(results["metrics_by_level"]) == 4


class TestDriftVisualization:
    """Test visualization functions."""

    def test_drift_visualization(self, reference_data, drifted_data, tmp_path):
        """Test drift analysis visualization."""
        result = DataDriftDetector.compare_distributions(reference_data, drifted_data)
        output_path = str(tmp_path / "drift_analysis.png")

        visualize_drift_analysis(result, output_path=output_path)

        assert Path(output_path).exists()

    def test_robustness_visualization(self, sample_data, test_model, model_device, tmp_path):
        """Test robustness analysis visualization."""
        x, y = sample_data
        analyzer = RobustnessAnalyzer(test_model, device=model_device)

        results = analyzer.evaluate_robustness(
            x, y, perturbation_type="noise", perturbation_levels=[0.0, 0.01, 0.05, 0.1]
        )
        output_path = str(tmp_path / "robustness_analysis.png")

        visualize_robustness_analysis(results, output_path=output_path)

        assert Path(output_path).exists()


class TestIntegrationDriftDetection:
    """Integration tests with real preprocessed data."""

    @pytest.mark.integration
    def test_drift_detection_with_preprocessed_data(self):
        """Test drift detection using real preprocessed dataset."""
        preprocessed_path = Path("data/preprocessed")
        if not preprocessed_path.exists():
            pytest.skip("Preprocessed data not available")

        # Load dataset
        dataset = MyDataset(
            data_path=Path("data/raw/ukdale.h5"),
            preprocessed_folder=preprocessed_path,
        )

        n = len(dataset)
        n_train = int(0.8 * n)
        n_drift_test = int(0.1 * n)
        n_normal_test = n - n_train - n_drift_test

        train_ds, drift_test_ds, normal_test_ds = torch.utils.data.random_split(
            dataset, [n_train, n_drift_test, n_normal_test], generator=torch.Generator().manual_seed(42)
        )

        # Extract mains power (input) from datasets
        train_mains = np.concatenate([dataset[i][0].numpy() for i in range(len(train_ds))])
        normal_test_mains = np.concatenate([dataset[i][0].numpy() for i in range(len(normal_test_ds))])

        # Create drift in test set (scale shift)
        drift_mains = normal_test_mains * 1.2  # 20% amplitude increase

        # Detect drift
        result_normal = DataDriftDetector.compare_distributions(
            train_mains.flatten(), normal_test_mains.flatten(), "normal_test"
        )
        result_drift = DataDriftDetector.compare_distributions(
            train_mains.flatten(), drift_mains.flatten(), "drifted_test"
        )

        # Normal test should have low drift
        assert result_normal["psi"] < 0.25

        # Drifted test should have high drift
        assert result_drift["psi"] > 0.25

        logger.info(f"Normal test PSI: {result_normal['psi']:.4f}")
        logger.info(f"Drifted test PSI: {result_drift['psi']:.4f}")

    @pytest.mark.integration
    def test_model_robustness_with_real_checkpoint(self):
        """Test model robustness with real trained checkpoint."""
        checkpoint_path = Path("models/best.pt")
        if not checkpoint_path.exists():
            pytest.skip("Model checkpoint not available")

        preprocessed_path = Path("data/preprocessed")
        if not preprocessed_path.exists():
            pytest.skip("Preprocessed data not available")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        model = Model().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # Load test data
        dataset = MyDataset(
            data_path=Path("data/raw/ukdale.h5"),
            preprocessed_folder=preprocessed_path,
        )

        n = len(dataset)
        n_train = int(0.9 * n)
        _, test_ds = torch.utils.data.random_split(
            dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(42)
        )

        loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        x_test, y_test = next(iter(loader))
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Test robustness
        analyzer = RobustnessAnalyzer(model, device=device)
        results = analyzer.evaluate_robustness(
            x_test, y_test, perturbation_type="noise", perturbation_levels=[0.0, 0.01, 0.05, 0.1]
        )

        # Model should maintain reasonable performance under small noise
        baseline_rmse = results["baseline_metric"]["rmse"]
        max_noise_rmse = results["metrics_by_level"][0.1]["rmse"]

        # Allow up to 2x degradation
        assert max_noise_rmse < 2 * baseline_rmse
        logger.info(f"Robustness: baseline RMSE={baseline_rmse:.6f}, at 0.1 noise RMSE={max_noise_rmse:.6f}")


# Import logger at module level for test output
from loguru import logger
