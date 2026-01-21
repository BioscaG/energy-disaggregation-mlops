"""
Data drift detection and robustness testing module.

This module provides tools to assess model robustness against data distribution shifts
(covariate drift, concept drift, and label drift) in energy disaggregation tasks.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DataDriftDetector:
    """
    Detects and measures data distribution drift using statistical tests.

    Methods:
    - Kolmogorov-Smirnov (KS) test
    - Population Stability Index (PSI)
    - Chi-square test (for distributions)
    - Maximum Mean Discrepancy (MMD)
    """

    @staticmethod
    def kolmogorov_smirnov_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """
        Performs KS test to detect distribution shift.

        Args:
            reference: Reference distribution (training data)
            current: Current distribution (test/production data)

        Returns:
            dict with statistic and p-value
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return {"statistic": float(statistic), "p_value": float(p_value), "drift_detected": bool(p_value < 0.05)}

    @staticmethod
    def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculates PSI to measure population stability.
        PSI > 0.25 indicates significant drift.

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram

        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))

        ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1e-10
        cur_counts = np.histogram(current, bins=breakpoints)[0] + 1e-10

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    @staticmethod
    def maximum_mean_discrepancy(reference: np.ndarray, current: np.ndarray, kernel: str = "rbf") -> float:
        """
        Computes Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            reference: Reference distribution
            current: Current distribution
            kernel: Kernel type ("rbf" or "linear")

        Returns:
            MMD value
        """
        # Flatten if needed
        X = reference.flatten() if reference.ndim > 1 else reference
        Y = current.flatten() if current.ndim > 1 else current

        if kernel == "rbf":
            # Use median heuristic for bandwidth
            all_data = np.concatenate([X, Y])
            # Compute pairwise distances
            X_expanded = all_data[:, np.newaxis]  # [n, 1]
            Y_expanded = all_data[np.newaxis, :]  # [1, n]
            pairwise_dists = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=0))
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])
            sigma = median_dist / np.sqrt(2)

            # RBF kernel
            def kernel_fn(a, b):
                return np.exp(-((a - b) ** 2) / (2 * sigma**2))
        else:
            # Linear kernel
            def kernel_fn(a, b):
                return a * b

        # Compute MMD
        n, m = len(X), len(Y)
        Kxx = kernel_fn(X[:, None], X[None, :]).mean()
        Kyy = kernel_fn(Y[:, None], Y[None, :]).mean()
        Kxy = kernel_fn(X[:, None], Y[None, :]).mean()

        mmd_sq = Kxx + Kyy - 2 * Kxy
        return float(np.sqrt(np.abs(mmd_sq)))

    @staticmethod
    def compare_distributions(
        reference: np.ndarray, current: np.ndarray, feature_name: str = "feature"
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical comparison of distributions.

        Returns:
            Dictionary with multiple drift metrics
        """
        return {
            "feature": feature_name,
            "reference_mean": float(np.mean(reference)),
            "reference_std": float(np.std(reference)),
            "current_mean": float(np.mean(current)),
            "current_std": float(np.std(current)),
            "mean_shift": float(np.abs(np.mean(current) - np.mean(reference))),
            "ks_test": DataDriftDetector.kolmogorov_smirnov_test(reference, current),
            "psi": DataDriftDetector.population_stability_index(reference, current),
            "mmd": DataDriftDetector.maximum_mean_discrepancy(reference, current),
        }


class RobustnessAnalyzer:
    """
    Analyzes model robustness against various data drift scenarios.

    Supports:
    - Additive noise injection
    - Scale/amplitude shifts
    - Frequency shifts
    - Missing data simulation
    - Distribution shift
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize analyzer.

        Args:
            model: PyTorch model to test
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.eval()

    def add_gaussian_noise(self, x: np.ndarray, noise_levels: List[float]) -> Dict[float, np.ndarray]:
        """
        Add Gaussian noise at different levels.

        Args:
            x: Input data [B, 1, T] or [B, T]
            noise_levels: List of noise standard deviations

        Returns:
            Dictionary mapping noise level to noisy data
        """
        noisy_data = {}
        for noise_std in noise_levels:
            noise = np.random.normal(0, noise_std, x.shape)
            noisy_data[noise_std] = x + noise
        return noisy_data

    def scale_amplitude(self, x: np.ndarray, scale_factors: List[float]) -> Dict[float, np.ndarray]:
        """
        Scale input amplitude (simulating different sensor scales).

        Args:
            x: Input data
            scale_factors: List of scaling factors

        Returns:
            Dictionary mapping scale factor to scaled data
        """
        scaled_data = {}
        for scale in scale_factors:
            scaled_data[scale] = x * scale
        return scaled_data

    def add_missing_data(self, x: np.ndarray, missing_rates: List[float]) -> Dict[float, np.ndarray]:
        """
        Simulate missing data by zero-filling random windows.

        Args:
            x: Input data
            missing_rates: List of missing data rates (0-1)

        Returns:
            Dictionary mapping missing rate to data with gaps
        """
        missing_data = {}
        for rate in missing_rates:
            x_missing = x.copy()
            T = x.shape[-1] if x.ndim > 1 else len(x)
            num_missing = int(T * rate)

            if num_missing > 0:
                # Create random missing windows
                missing_indices = np.random.choice(T, num_missing, replace=False)
                if x.ndim == 3:  # [B, 1, T]
                    x_missing[:, :, missing_indices] = 0
                else:  # [B, T]
                    x_missing[:, missing_indices] = 0

            missing_data[rate] = x_missing
        return missing_data

    def evaluate_robustness(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        perturbation_type: str = "noise",
        perturbation_levels: List[float] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Evaluate model robustness under perturbations.

        Args:
            x_test: Test input [B, 1, T]
            y_test: Test target [B, 1, T]
            perturbation_type: Type of perturbation ("noise", "scale", "missing")
            perturbation_levels: Levels to test
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with robustness metrics
        """
        if perturbation_levels is None:
            if perturbation_type == "noise":
                perturbation_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
            elif perturbation_type == "scale":
                perturbation_levels = [0.5, 0.8, 1.0, 1.2, 1.5]
            elif perturbation_type == "missing":
                perturbation_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

        x_np = x_test.cpu().numpy()
        y_np = y_test.cpu().numpy()

        # Generate perturbed data
        if perturbation_type == "noise":
            perturbed_data = self.add_gaussian_noise(x_np, perturbation_levels)
        elif perturbation_type == "scale":
            perturbed_data = self.scale_amplitude(x_np, perturbation_levels)
        elif perturbation_type == "missing":
            perturbed_data = self.add_missing_data(x_np, perturbation_levels)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        results = {"perturbation_type": perturbation_type, "baseline_metric": None, "metrics_by_level": {}}

        criterion = nn.MSELoss()

        with torch.no_grad():
            for level, x_perturbed in perturbed_data.items():
                x_perturbed_tensor = torch.from_numpy(x_perturbed).float().to(self.device)
                dataset = TensorDataset(x_perturbed_tensor, y_test)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                total_mse = 0.0
                total_mae = 0.0
                num_samples = 0

                for x_batch, y_batch in loader:
                    y_pred = self.model(x_batch)
                    mse = criterion(y_pred, y_batch).item()
                    mae = torch.abs(y_pred - y_batch).mean().item()

                    total_mse += mse * x_batch.size(0)
                    total_mae += mae * x_batch.size(0)
                    num_samples += x_batch.size(0)

                results["metrics_by_level"][level] = {
                    "mse": total_mse / num_samples,
                    "rmse": np.sqrt(total_mse / num_samples),
                    "mae": total_mae / num_samples,
                }

                if level == (
                    perturbation_levels[0]
                    if perturbation_type == "noise"
                    else (1.0 if perturbation_type == "scale" else 0.0)
                ):
                    results["baseline_metric"] = results["metrics_by_level"][level]

        return results


def visualize_drift_analysis(drift_results: Dict[str, Any], output_path: str = "drift_analysis.png"):
    """
    Visualize drift detection results.

    Args:
        drift_results: Results from drift detection
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if isinstance(drift_results, dict) and "ks_test" in drift_results:
        # Single feature drift analysis
        ax = axes[0, 0]
        ax.text(
            0.1,
            0.5,
            f"KS Statistic: {drift_results['ks_test']['statistic']:.4f}\n"
            f"P-value: {drift_results['ks_test']['p_value']:.4f}\n"
            f"Drift Detected: {drift_results['ks_test']['drift_detected']}",
            fontsize=12,
        )
        ax.set_title("Kolmogorov-Smirnov Test")
        ax.axis("off")

        ax = axes[0, 1]
        ax.text(
            0.1,
            0.5,
            f"PSI: {drift_results['psi']:.4f}\n(Threshold: 0.25)\nSignificant: {drift_results['psi'] > 0.25}",
            fontsize=12,
        )
        ax.set_title("Population Stability Index")
        ax.axis("off")

        ax = axes[1, 0]
        ax.text(0.1, 0.5, f"MMD: {drift_results['mmd']:.4f}", fontsize=12)
        ax.set_title("Maximum Mean Discrepancy")
        ax.axis("off")

        ax = axes[1, 1]
        ax.text(
            0.1,
            0.5,
            f"Reference Mean: {drift_results['reference_mean']:.4f}\n"
            f"Current Mean: {drift_results['current_mean']:.4f}\n"
            f"Mean Shift: {drift_results['mean_shift']:.4f}",
            fontsize=12,
        )
        ax.set_title("Distribution Statistics")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    logger.success(f"Drift analysis visualization saved to {output_path}")


def visualize_robustness_analysis(robustness_results: Dict[str, Any], output_path: str = "robustness_analysis.png"):
    """
    Visualize robustness testing results.

    Args:
        robustness_results: Results from robustness analyzer
        output_path: Path to save visualization
    """
    perturbation_type = robustness_results["perturbation_type"]
    metrics = robustness_results["metrics_by_level"]

    levels = sorted(metrics.keys())
    rmse_values = [metrics[l]["rmse"] for l in levels]
    mae_values = [metrics[l]["mae"] for l in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(levels, rmse_values, marker="o", linewidth=2, markersize=8, label="RMSE")
    ax1.set_xlabel(f"{perturbation_type.capitalize()} Level")
    ax1.set_ylabel("RMSE")
    ax1.set_title(f"Model Robustness to {perturbation_type.capitalize()}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(levels, mae_values, marker="s", linewidth=2, markersize=8, color="orange", label="MAE")
    ax2.set_xlabel(f"{perturbation_type.capitalize()} Level")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"Model Robustness to {perturbation_type.capitalize()}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    logger.success(f"Robustness analysis visualization saved to {output_path}")
