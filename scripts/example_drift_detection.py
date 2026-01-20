#!/usr/bin/env python
"""
Quick example script demonstrating drift detection and robustness testing.
Run this script to see the module in action with synthetic data.
"""

from pathlib import Path

import numpy as np
import torch

from energy_dissagregation_mlops.drift_detection import (
    DataDriftDetector,
    RobustnessAnalyzer,
    visualize_drift_analysis,
    visualize_robustness_analysis,
)
from energy_dissagregation_mlops.model import Model


def example_drift_detection():
    """Example 1: Detect data drift"""
    print("=" * 70)
    print("EXAMPLE 1: DATA DRIFT DETECTION")
    print("=" * 70)

    # Generate synthetic training data (baseline)
    np.random.seed(42)
    reference = np.random.normal(loc=500, scale=50, size=1000)
    print(f"\nBaseline distribution: mean={reference.mean():.1f}, std={reference.std():.1f}")

    # Scenario 1: Normal test set (no drift)
    print("\n--- Scenario 1: No Drift (Normal Test Set) ---")
    test_no_drift = np.random.normal(loc=500, scale=50, size=1000)
    result = DataDriftDetector.compare_distributions(reference, test_no_drift)
    print(f"KS p-value: {result['ks_test']['p_value']:.6f} (drift: {result['ks_test']['drift_detected']})")
    print(f"PSI: {result['psi']:.4f} (threshold: 0.25)")
    print(f"MMD: {result['mmd']:.4f}")

    # Scenario 2: Drifted data (mean shift)
    print("\n--- Scenario 2: Drift Detected (15% Mean Shift) ---")
    test_drifted = np.random.normal(loc=575, scale=50, size=1000)  # 15% increase
    result = DataDriftDetector.compare_distributions(reference, test_drifted)
    print(f"KS p-value: {result['ks_test']['p_value']:.6f} (drift: {result['ks_test']['drift_detected']})")
    print(f"PSI: {result['psi']:.4f} (SIGNIFICANT)" if result["psi"] > 0.25 else f"PSI: {result['psi']:.4f}")
    print(f"MMD: {result['mmd']:.4f}")

    # Save visualization
    visualize_drift_analysis(result, output_path="/tmp/drift_example.png")
    print("\n✅ Visualization saved to /tmp/drift_example.png")


def example_robustness_testing():
    """Example 2: Test model robustness"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: MODEL ROBUSTNESS TESTING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model
    model = Model()
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Generate test data
    x_test = torch.randn(100, 1, 1024, device=device)  # [B, C, T]
    y_test = torch.randn(100, 1, 1024, device=device)
    print(f"Test data shape: x={x_test.shape}, y={y_test.shape}")

    # Test robustness to noise
    print("\n--- Testing Robustness to Gaussian Noise ---")
    analyzer = RobustnessAnalyzer(model, device=device)
    results = analyzer.evaluate_robustness(
        x_test, y_test, perturbation_type="noise", perturbation_levels=[0.0, 0.01, 0.05, 0.1], batch_size=32
    )

    baseline_rmse = results["baseline_metric"]["rmse"]
    print(f"Baseline RMSE: {baseline_rmse:.6f}")

    for level in sorted(results["metrics_by_level"].keys()):
        metrics = results["metrics_by_level"][level]
        degradation = (metrics["rmse"] - baseline_rmse) / baseline_rmse * 100
        print(f"  Noise level {level}: RMSE={metrics['rmse']:.6f} (degradation: {degradation:+.1f}%)")

    # Classify robustness
    worst_rmse = max(m["rmse"] for m in results["metrics_by_level"].values())
    degradation = (worst_rmse - baseline_rmse) / baseline_rmse * 100

    if degradation < 20:
        status = "✅ ROBUST"
    elif degradation < 50:
        status = "⚠️ MODERATE"
    else:
        status = "❌ LOW ROBUSTNESS"

    print(f"\n🎯 Robustness Classification: {status}")
    print(f"   Max degradation: {degradation:.1f}%")

    # Save visualization
    visualize_robustness_analysis(results, output_path="/tmp/robustness_example.png")
    print("✅ Visualization saved to /tmp/robustness_example.png")


def example_comparison():
    """Example 3: Compare multiple perturbation types"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: COMPARING PERTURBATION TYPES")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model()
    model = model.to(device)
    model.eval()

    # Generate test data
    x_test = torch.randn(50, 1, 1024, device=device)
    y_test = torch.randn(50, 1, 1024, device=device)

    analyzer = RobustnessAnalyzer(model, device=device)

    perturbation_types = [
        ("noise", [0.0, 0.05, 0.1]),
        ("scale", [0.8, 1.0, 1.2]),
        ("missing", [0.0, 0.1, 0.2]),
    ]

    print("\nTesting model robustness across different perturbations:\n")

    for pert_type, levels in perturbation_types:
        results = analyzer.evaluate_robustness(
            x_test, y_test, perturbation_type=pert_type, perturbation_levels=levels, batch_size=32
        )

        baseline_rmse = results["baseline_metric"]["rmse"]
        worst_rmse = max(m["rmse"] for m in results["metrics_by_level"].values())
        degradation = (worst_rmse - baseline_rmse) / baseline_rmse * 100

        print(f"{pert_type.upper()}:")
        print(f"  Baseline RMSE: {baseline_rmse:.6f}")
        print(f"  Worst-case RMSE: {worst_rmse:.6f}")
        print(f"  Degradation: {degradation:.1f}%")
        print()


if __name__ == "__main__":
    # Run examples
    example_drift_detection()
    example_robustness_testing()
    example_comparison()

    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nFor more details, see: docs/DRIFT_DETECTION.md")
    print("To run the full analysis: python scripts/test_drift_robustness.py")
