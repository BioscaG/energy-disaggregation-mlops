#!/usr/bin/env python
"""
Comprehensive drift detection and robustness analysis script (M27).

This script runs a complete evaluation of model robustness against data drift:
1. Tests multiple types of drifts (scale, noise, missing data)
2. Uses multiple statistical tests (KS, PSI, MMD)
3. Evaluates model performance degradation
4. Generates detailed reports and visualizations
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from energy_dissagregation_mlops.data import MyDataset
from energy_dissagregation_mlops.drift_detection import (
    DataDriftDetector,
    RobustnessAnalyzer,
    visualize_drift_analysis,
    visualize_robustness_analysis,
)
from energy_dissagregation_mlops.model import Model


def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "drift_analysis.log"

    logger.remove()  # Remove default handler
    logger.add(
        sink=log_file,
        format=(
            "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level="INFO",
    )
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=(
            "<level>[{time:HH:mm:ss}]</level> <level>{level: <8}</level> - "
            "<level>{message}</level>"
        ),
        level="INFO",
    )

    return log_file



def load_model_and_data(
    checkpoint_path: Path, data_path: Path, preprocessed_folder: Path, device: str, num_test_samples: int = 100
) -> tuple:
    """Load model and test data."""
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING MODEL AND DATA")
    logger.info("=" * 70)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logger.success(f"Model loaded on {device}")

    # Load dataset
    logger.info(f"Loading dataset from {data_path}")
    dataset = MyDataset(data_path=data_path, preprocessed_folder=preprocessed_folder)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Split into train and test
    n = len(dataset)
    n_train = int(0.9 * n)
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(42)
    )

    # Sample test data for efficiency
    indices = np.random.choice(len(test_ds), min(num_test_samples, len(test_ds)), replace=False)
    test_ds_sampled = Subset(test_ds, indices)

    logger.info(f"Training set size: {len(train_ds)}")
    logger.info(f"Test set size: {len(test_ds_sampled)}")

    return model, train_ds, test_ds_sampled, device


def run_drift_detection(train_ds, test_ds, output_dir: Path) -> Dict[str, Any]:
    """Run comprehensive drift detection analysis."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DATA DRIFT DETECTION")
    logger.info("=" * 70)

    # Extract mains power
    logger.info("Extracting mains power from training set...")
    train_mains = []
    for idx in range(min(len(train_ds), 200)):
        x, _ = train_ds[idx]
        train_mains.append(x.numpy().flatten())
    train_mains = np.concatenate(train_mains)

    logger.info("Extracting mains power from test set...")
    test_mains = []
    for idx in range(min(len(test_ds), 200)):
        x, _ = test_ds[idx]
        test_mains.append(x.numpy().flatten())
    test_mains = np.concatenate(test_mains)

    logger.info(f"Train mains: mean={train_mains.mean():.2f}, std={train_mains.std():.2f}")
    logger.info(f"Test mains: mean={test_mains.mean():.2f}, std={test_mains.std():.2f}")

    # No drift (baseline)
    logger.info("\n--- Testing BASELINE (no drift) ---")
    result_baseline = DataDriftDetector.compare_distributions(train_mains, test_mains)

    drift_results = {"baseline": result_baseline, "drift_scenarios": {}}

    # Test different drift scenarios
    drift_scenarios = [
        ("scale_10pct", lambda x: x * 1.1, "10% amplitude increase"),
        ("scale_20pct", lambda x: x * 1.2, "20% amplitude increase"),
        ("noise_5pct", lambda x: x + np.random.normal(0, 0.05 * x.std(), x.shape), "5% noise"),
        ("noise_10pct", lambda x: x + np.random.normal(0, 0.1 * x.std(), x.shape), "10% noise"),
        ("missing_10pct", lambda x: _apply_missing_data(x, 0.1), "10% missing data"),
        ("missing_20pct", lambda x: _apply_missing_data(x, 0.2), "20% missing data"),
    ]

    for scenario_name, drift_fn, description in drift_scenarios:
        logger.info(f"\n--- Testing {description} ---")
        drifted_mains = drift_fn(test_mains)
        result = DataDriftDetector.compare_distributions(train_mains, drifted_mains, scenario_name)
        drift_results["drift_scenarios"][scenario_name] = result

        logger.info(f"KS p-value: {result['ks_test']['p_value']:.6f} (drift: {result['ks_test']['drift_detected']})")
        logger.info(f"PSI: {result['psi']:.4f} (threshold: 0.25)")
        logger.info(f"MMD: {result['mmd']:.4f}")

    # Save drift results
    drift_results_file = output_dir / "drift_detection_results.json"
    _save_drift_results_to_json(drift_results, drift_results_file)

    # Visualize baseline
    visualize_drift_analysis(result_baseline, output_path=str(output_dir / "drift_baseline.png"))

    logger.success("Drift detection complete!")
    return drift_results


def run_robustness_analysis(model, test_ds, device: str, output_dir: Path, batch_size: int = 32) -> Dict[str, Any]:
    """Run model robustness testing."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: MODEL ROBUSTNESS TESTING")
    logger.info("=" * 70)

    # Prepare test data
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    x_test_list, y_test_list = [], []
    for x, y in loader:
        x_test_list.append(x)
        y_test_list.append(y)

    x_test = torch.cat(x_test_list).to(device)
    y_test = torch.cat(y_test_list).to(device)

    logger.info(f"Test data shape: x={x_test.shape}, y={y_test.shape}")

    analyzer = RobustnessAnalyzer(model, device=device)
    robustness_results = {}

    # Test different perturbation types
    perturbations = [
        ("noise", [0.0, 0.005, 0.01, 0.02, 0.05]),
        ("scale", [0.8, 0.9, 1.0, 1.1, 1.2]),
        ("missing", [0.0, 0.05, 0.1, 0.15, 0.2]),
    ]

    for perturbation_type, levels in perturbations:
        logger.info(f"\n--- Testing {perturbation_type} robustness ---")
        results = analyzer.evaluate_robustness(
            x_test, y_test, perturbation_type=perturbation_type, perturbation_levels=levels, batch_size=batch_size
        )

        robustness_results[perturbation_type] = results

        # Log results
        baseline_rmse = results["baseline_metric"]["rmse"]
        logger.info(f"Baseline RMSE: {baseline_rmse:.6f}")

        for level in sorted(results["metrics_by_level"].keys()):
            metrics = results["metrics_by_level"][level]
            degradation = ((metrics["rmse"] - baseline_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0
            logger.info(f"  Level {level}: RMSE={metrics['rmse']:.6f} (degradation: {degradation:+.1f}%)")

        # Visualize
        visualize_robustness_analysis(results, output_path=str(output_dir / f"robustness_{perturbation_type}.png"))

    logger.success("Robustness analysis complete!")
    return robustness_results


def generate_report(drift_results: Dict, robustness_results: Dict, output_dir: Path):
    """Generate comprehensive analysis report."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: GENERATING REPORT")
    logger.info("=" * 70)

    report = []
    report.append("# Data Drift Robustness Analysis Report (M27)\n")

    # Drift Detection Section
    report.append("## 1. Data Drift Detection\n")

    baseline = drift_results["baseline"]
    report.append("### Baseline Distribution Comparison (No Drift)\n")
    report.append(f"- KS Statistic: {baseline['ks_test']['statistic']:.4f}\n")
    report.append(f"- P-value: {baseline['ks_test']['p_value']:.6f}\n")
    report.append(f"- PSI: {baseline['psi']:.4f}\n")
    report.append(f"- MMD: {baseline['mmd']:.4f}\n")
    report.append(f"- Mean shift: {baseline['mean_shift']:.4f}\n")
    report.append("\n")

    report.append("### Drift Scenarios\n")
    report.append("| Scenario | KS p-value | PSI | MMD | Drift Detected |\n")
    report.append("|----------|-----------|-----|-----|----------------|\n")

    for scenario, result in drift_results["drift_scenarios"].items():
        report.append(
            f"| {scenario} | {result['ks_test']['p_value']:.6f} | "
            f"{result['psi']:.4f} | {result['mmd']:.4f} | "
            f"{result['ks_test']['drift_detected']} |\n"
        )
    report.append("\n")

    # Robustness Section
    report.append("## 2. Model Robustness Testing\n")

    for pert_type, results in robustness_results.items():
        report.append(f"### {pert_type.capitalize()} Robustness\n")

        baseline_rmse = results["baseline_metric"]["rmse"]
        report.append(f"- Baseline RMSE: {baseline_rmse:.6f}\n")

        report.append("\n| Level | RMSE | MAE | Degradation |\n")
        report.append("|-------|------|-----|-------------|\n")

        for level in sorted(results["metrics_by_level"].keys()):
            metrics = results["metrics_by_level"][level]
            degradation = ((metrics["rmse"] - baseline_rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0
            report.append(f"| {level} | {metrics['rmse']:.6f} | {metrics['mae']:.6f} | {degradation:+.1f}% |\n")

        report.append("\n")

    # Summary
    report.append("## 3. Summary\n")

    # Check drift detection
    significant_drifts = sum(1 for r in drift_results["drift_scenarios"].values() if r["psi"] > 0.25)
    report.append(
        f"- Significant drifts detected (PSI > 0.25): {significant_drifts}/{len(drift_results['drift_scenarios'])}\n"
    )

    # Check robustness
    for pert_type, results in robustness_results.items():
        baseline_rmse = results["baseline_metric"]["rmse"]
        worst_rmse = max(m["rmse"] for m in results["metrics_by_level"].values())
        degradation = (worst_rmse - baseline_rmse) / baseline_rmse * 100

        if degradation < 20:
            status = "✅ ROBUST"
        elif degradation < 50:
            status = "⚠️ MODERATE"
        else:
            status = "❌ LOW ROBUSTNESS"

        report.append(f"- {pert_type.capitalize()} robustness: {status} ({degradation:.1f}% degradation)\n")

    report.append("\n## 4. Visualizations\n")
    report.append("- `drift_baseline.png`: Baseline drift detection results\n")
    report.append("- `robustness_noise.png`: Noise perturbation robustness\n")
    report.append("- `robustness_scale.png`: Scale perturbation robustness\n")
    report.append("- `robustness_missing.png`: Missing data robustness\n")

    # Save report
    report_file = output_dir / "DRIFT_REPORT.md"
    with open(report_file, "w") as f:
        f.writelines(report)

    logger.success(f"Report saved to {report_file}")

    # Print report to console
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    print("".join(report))


def _apply_missing_data(x: np.ndarray, missing_rate: float) -> np.ndarray:
    """Apply missing data by zero-filling random samples."""
    x_missing = x.copy()
    num_missing = int(len(x) * missing_rate)
    if num_missing > 0:
        missing_idx = np.random.choice(len(x), num_missing, replace=False)
        x_missing[missing_idx] = 0
    return x_missing


def _save_drift_results_to_json(results: Dict, filepath: Path):
    """Save drift results to JSON for storage."""

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {filepath}")


def main(
    checkpoint_path: str = "models/best.pt",
    data_path: str = "data/raw/ukdale.h5",
    preprocessed_folder: str = "data/preprocessed",
    output_dir: str = "drift_analysis_output",
    num_test_samples: int = 100,
):
    """
    Run comprehensive drift detection and robustness analysis.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to raw dataset
        preprocessed_folder: Path to preprocessed data
        output_dir: Output directory for results
        num_test_samples: Number of test samples to use
    """
    output_dir = Path(output_dir)
    setup_logging(output_dir)

    logger.info("🚀 STARTING COMPREHENSIVE DRIFT DETECTION ANALYSIS (M27)")
    logger.info(f"Output directory: {output_dir}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Step 1: Load model and data
        model, train_ds, test_ds, device = load_model_and_data(
            Path(checkpoint_path), Path(data_path), Path(preprocessed_folder), device, num_test_samples
        )

        # Step 2: Run drift detection
        drift_results = run_drift_detection(train_ds, test_ds, output_dir)

        # Step 3: Run robustness analysis
        robustness_results = run_robustness_analysis(model, test_ds, device, output_dir)

        # Step 4: Generate report
        generate_report(drift_results, robustness_results, output_dir)

        logger.success("✅ Analysis complete! All results saved to output directory.")

    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    import sys

    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kwargs[key] = value

    main(**kwargs)
