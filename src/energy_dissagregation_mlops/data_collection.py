"""
Data collection from deployed application for drift monitoring.

Collects input-output data from predictions to enable:
- Performance monitoring
- Data drift detection
- Model retraining triggers
"""

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger


class ProductionDataCollector:
    """Collects and manages production prediction data."""

    def __init__(self, collection_dir: str = "data/production", max_samples: int = 10000):
        """
        Initialize collector.

        Args:
            collection_dir: Directory to store collected data
            max_samples: Maximum samples to keep in memory
        """
        self.collection_dir = Path(collection_dir)
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

        # In-memory buffers
        self.predictions = []
        self.inputs = []
        self.outputs = []
        self.timestamps = []
        self.metadata = []

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.total_predictions = 0
        self.errors_count = 0

        logger.info(f"ProductionDataCollector initialized at {self.collection_dir}")

    def record_prediction(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        prediction_id: str = None,
        metadata: Dict[str, Any] = None,
        error: str = None,
    ):
        """
        Record a single prediction.

        Args:
            input_data: Input features (mains power)
            output_data: Model output (predicted appliance power)
            prediction_id: Unique prediction ID
            metadata: Additional metadata
            error: Error message if prediction failed
        """
        with self.lock:
            timestamp = datetime.utcnow().isoformat()

            if error:
                self.errors_count += 1
                logger.warning(f"Prediction error recorded: {error}")
                return

            # Convert to float for JSON serialization
            input_flat = input_data.flatten().astype(float)
            output_flat = output_data.flatten().astype(float)

            self.inputs.append(input_flat.tolist())
            self.outputs.append(output_flat.tolist())
            self.timestamps.append(timestamp)

            meta = metadata or {}
            meta["prediction_id"] = prediction_id
            self.metadata.append(meta)

            self.total_predictions += 1

            # Maintain max size
            if len(self.inputs) > self.max_samples:
                self.inputs.pop(0)
                self.outputs.pop(0)
                self.timestamps.pop(0)
                self.metadata.pop(0)

            if self.total_predictions % 100 == 0:
                logger.info(f"Collected {self.total_predictions} predictions")

    def save_batch(self, filename: str = None) -> Path:
        """
        Save collected data to disk.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        with self.lock:
            if not self.inputs:
                logger.warning("No data to save")
                return None

            if filename is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions_{timestamp}.json"

            filepath = self.collection_dir / filename

            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "num_predictions": len(self.inputs),
                "predictions": [
                    {
                        "input": inp,
                        "output": out,
                        "timestamp": ts,
                        "metadata": meta,
                    }
                    for inp, out, ts, meta in zip(self.inputs, self.outputs, self.timestamps, self.metadata)
                ],
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.success(f"Saved {len(self.inputs)} predictions to {filepath}")
            return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        with self.lock:
            if not self.inputs:
                return {
                    "total_predictions": 0,
                    "buffered_predictions": 0,
                    "errors_count": self.errors_count,
                    "status": "No data collected yet",
                }

            inputs_array = np.array(self.inputs)
            outputs_array = np.array(self.outputs)

            return {
                "total_predictions": self.total_predictions,
                "buffered_predictions": len(self.inputs),
                "errors_count": self.errors_count,
                "input_statistics": {
                    "mean": float(inputs_array.mean()),
                    "std": float(inputs_array.std()),
                    "min": float(inputs_array.min()),
                    "max": float(inputs_array.max()),
                },
                "output_statistics": {
                    "mean": float(outputs_array.mean()),
                    "std": float(outputs_array.std()),
                    "min": float(outputs_array.min()),
                    "max": float(outputs_array.max()),
                },
                "time_range": {
                    "first": self.timestamps[0] if self.timestamps else None,
                    "last": self.timestamps[-1] if self.timestamps else None,
                },
            }

    def get_recent_data(self, n: int = 100) -> Dict[str, Any]:
        """Get recent n predictions."""
        with self.lock:
            recent_n = min(n, len(self.inputs))
            return {
                "count": recent_n,
                "predictions": [
                    {
                        "input": inp,
                        "output": out,
                        "timestamp": ts,
                    }
                    for inp, out, ts in zip(
                        self.inputs[-recent_n:],
                        self.outputs[-recent_n:],
                        self.timestamps[-recent_n:],
                    )
                ],
            }

    def get_hourly_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get hourly summary of predictions."""
        with self.lock:
            if not self.timestamps:
                return {}

            hourly = defaultdict(list)
            for inp, out, ts in zip(self.inputs, self.outputs, self.timestamps):
                # Extract hour from ISO timestamp
                hour = ts[:13] + ":00:00"  # YYYY-MM-DDTHH:00:00
                hourly[hour].append({"input": inp, "output": out})

            summary = {}
            for hour, data in sorted(hourly.items()):
                inputs_array = np.array([d["input"] for d in data])
                outputs_array = np.array([d["output"] for d in data])

                summary[hour] = {
                    "count": len(data),
                    "input_mean": float(inputs_array.mean()),
                    "output_mean": float(outputs_array.mean()),
                }

            return summary

    def clear(self):
        """Clear all collected data from memory."""
        with self.lock:
            self.inputs.clear()
            self.outputs.clear()
            self.timestamps.clear()
            self.metadata.clear()
            logger.info("Cleared all collected data")

    def load_from_file(self, filepath: Path) -> Dict[str, Any]:
        """Load previously saved predictions."""
        with open(filepath) as f:
            data = json.load(f)

        for pred in data["predictions"]:
            self.inputs.append(pred["input"])
            self.outputs.append(pred["output"])
            self.timestamps.append(pred["timestamp"])
            self.metadata.append(pred["metadata"])

        logger.info(f"Loaded {len(data['predictions'])} predictions from {filepath}")
        return data


class DriftMonitor:
    """Monitor collected data for drift."""

    def __init__(self, collector: ProductionDataCollector):
        """
        Initialize drift monitor.

        Args:
            collector: ProductionDataCollector instance
        """
        self.collector = collector

    def analyze_drift(self, reference_dist: np.ndarray) -> Dict[str, Any]:
        """
        Analyze collected data for drift against reference.

        Args:
            reference_dist: Reference distribution from training

        Returns:
            Drift analysis results
        """
        stats = self.collector.get_statistics()

        if stats.get("buffered_predictions", 0) < 10:
            return {
                "status": "insufficient_data",
                "message": "Need at least 10 predictions for drift analysis",
                "current_count": stats.get("buffered_predictions", 0),
            }

        # Ensure we have data
        if not self.collector.inputs:
            return {
                "status": "insufficient_data",
                "message": "No predictions collected yet",
                "current_count": 0,
            }

        # Get current production data distribution
        try:
            from energy_dissagregation_mlops.drift_detection import DataDriftDetector

            current_inputs = np.array(self.collector.inputs).flatten()
            result = DataDriftDetector.compare_distributions(reference_dist, current_inputs)
        except Exception as e:
            logger.error(f"Error analyzing drift: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

        return {
            "status": "analyzed",
            "drift_detected": result["ks_test"]["drift_detected"],
            "psi": result["psi"],
            "ks_statistic": result["ks_test"]["statistic"],
            "ks_p_value": result["ks_test"]["p_value"],
            "mmd": result["mmd"],
            "interpretation": (
                "🚨 CRITICAL DRIFT"
                if result["psi"] > 0.25
                else "⚠️ MODERATE DRIFT"
                if result["psi"] > 0.1
                else "✅ STABLE"
            ),
            "recommendation": (
                "TRIGGER RETRAINING"
                if result["psi"] > 0.25
                else "INCREASE MONITORING"
                if result["psi"] > 0.1
                else "CONTINUE NORMAL OPERATION"
            ),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from collected data."""
        if not self.collector.outputs:
            return {"status": "no_data"}

        import torch
        from torch import nn

        # Compare predictions to idealized values
        # (This is a placeholder - in reality you'd have ground truth)
        outputs_array = np.array(self.collector.outputs).flatten()

        return {
            "num_predictions": len(self.collector.outputs),
            "output_mean": float(outputs_array.mean()),
            "output_std": float(outputs_array.std()),
            "output_min": float(outputs_array.min()),
            "output_max": float(outputs_array.max()),
        }


# Global instance
_collector = None


def get_collector() -> ProductionDataCollector:
    """Get or create global collector instance."""
    global _collector
    if _collector is None:
        _collector = ProductionDataCollector()
    return _collector


def get_drift_monitor() -> DriftMonitor:
    """Get drift monitor instance."""
    return DriftMonitor(get_collector())
