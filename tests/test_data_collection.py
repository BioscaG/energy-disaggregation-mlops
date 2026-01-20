"""
Tests for production data collection and monitoring.
"""

from pathlib import Path

import numpy as np
import pytest

from energy_dissagregation_mlops.data_collection import (
    DriftMonitor,
    ProductionDataCollector,
)


@pytest.fixture
def collector():
    """Create a test collector."""
    return ProductionDataCollector(
        collection_dir="/tmp/test_collection",
        max_samples=1000,
    )


@pytest.fixture
def reference_dist():
    """Create reference distribution."""
    np.random.seed(42)
    return np.random.normal(loc=500, scale=50, size=1000)


class TestProductionDataCollector:
    """Test data collection functionality."""

    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.total_predictions == 0
        assert collector.errors_count == 0
        assert len(collector.inputs) == 0

    def test_record_single_prediction(self, collector):
        """Test recording a single prediction."""
        x = np.random.randn(1024)
        y = np.random.randn(1024)

        collector.record_prediction(x, y, prediction_id="test_001")

        assert collector.total_predictions == 1
        assert len(collector.inputs) == 1
        assert len(collector.outputs) == 1

    def test_record_multiple_predictions(self, collector):
        """Test recording multiple predictions."""
        for i in range(10):
            x = np.random.randn(1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y, prediction_id=f"test_{i:03d}")

        assert collector.total_predictions == 10
        assert len(collector.inputs) == 10

    def test_max_samples_limit(self, collector):
        """Test that max samples limit is respected."""
        # Add more than max_samples
        for i in range(collector.max_samples + 100):
            x = np.random.randn(1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        # Should only keep max_samples
        assert len(collector.inputs) == collector.max_samples

    def test_record_error(self, collector):
        """Test recording an error."""
        collector.record_prediction(None, None, error="Test error message")

        assert collector.errors_count == 1
        assert collector.total_predictions == 0  # Error doesn't count

    def test_get_statistics(self, collector):
        """Test getting statistics."""
        # Add some data
        for i in range(5):
            x = np.random.randn(1024) + 500
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        stats = collector.get_statistics()

        assert stats["total_predictions"] == 5
        assert stats["buffered_predictions"] == 5
        assert "input_statistics" in stats
        assert "output_statistics" in stats

    def test_get_recent_data(self, collector):
        """Test getting recent predictions."""
        for i in range(10):
            x = np.random.randn(1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        recent = collector.get_recent_data(n=5)

        assert recent["count"] == 5
        assert len(recent["predictions"]) == 5

    def test_save_batch(self, collector, tmp_path):
        """Test saving batch to file."""
        collector.collection_dir = tmp_path

        for i in range(3):
            x = np.random.randn(1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        filepath = collector.save_batch("test_batch.json")

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_clear(self, collector):
        """Test clearing collector."""
        for i in range(5):
            x = np.random.randn(1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        assert collector.total_predictions == 5

        collector.clear()

        assert len(collector.inputs) == 0
        assert len(collector.outputs) == 0

    def test_get_hourly_summary(self, collector):
        """Test getting hourly summary."""
        for i in range(10):
            x = np.random.randn(1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        summary = collector.get_hourly_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0
        for hour, data in summary.items():
            assert "count" in data
            assert data["count"] > 0


class TestDriftMonitor:
    """Test drift monitoring functionality."""

    def test_monitor_initialization(self, collector):
        """Test monitor initialization."""
        monitor = DriftMonitor(collector)
        assert monitor.collector is collector

    def test_analyze_drift_insufficient_data(self, collector, reference_dist):
        """Test drift analysis with insufficient data."""
        # Add only 5 predictions (threshold is 10)
        for i in range(5):
            x = np.random.normal(500, 50, 1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        monitor = DriftMonitor(collector)
        result = monitor.analyze_drift(reference_dist)

        assert result["status"] == "insufficient_data"

    def test_analyze_drift_no_drift(self, collector, reference_dist):
        """Test drift analysis when there is no drift."""
        # Add predictions with same distribution as reference
        for i in range(20):
            x = np.random.normal(500, 50, 1024)
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        monitor = DriftMonitor(collector)
        result = monitor.analyze_drift(reference_dist)

        assert result["status"] == "analyzed"
        assert "psi" in result
        assert "drift_detected" in result

    def test_analyze_drift_with_drift(self, collector, reference_dist):
        """Test drift analysis when there is drift."""
        # Add predictions with different distribution (mean shifted up)
        for i in range(20):
            x = np.random.normal(600, 50, 1024)  # Shifted mean
            y = np.random.randn(1024)
            collector.record_prediction(x, y)

        monitor = DriftMonitor(collector)
        result = monitor.analyze_drift(reference_dist)

        assert result["status"] == "analyzed"
        # Should detect drift (PSI > 0.25)
        assert result["psi"] > 0.1 or result["drift_detected"]

    def test_get_performance_metrics(self, collector):
        """Test getting performance metrics."""
        for i in range(10):
            x = np.random.randn(1024)
            y = np.random.randn(1024) + 100
            collector.record_prediction(x, y)

        monitor = DriftMonitor(collector)
        metrics = monitor.get_performance_metrics()

        assert "num_predictions" in metrics
        assert metrics["num_predictions"] == 10
        assert "output_mean" in metrics
        assert "output_std" in metrics
