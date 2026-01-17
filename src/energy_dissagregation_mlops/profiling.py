"""
Profiling utilities for optimizing training and data loading performance.

Includes:
- PyTorch profiler for GPU/CPU profiling
- cProfile for Python function profiling
- Memory profiling
- Training speed metrics
"""

import cProfile
import pstats
import io
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import torch.profiler


class TrainingProfiler:
    """Profile training loop performance with detailed metrics."""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}
        self.timings: Dict[str, list] = {}

    @contextmanager
    def profile_section(self, name: str):
        """Context manager to profile a code section."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    def record_metric(self, name: str, value: Any):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all profiled sections."""
        import numpy as np

        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "total": float(np.sum(times)),
                "count": len(times),
            }
        return summary

    def save_results(self, filename: str = "profile_summary.json"):
        """Save profiling results to JSON."""
        summary = self.get_summary()
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "metrics": {k: list(v) if isinstance(v, list) else v for k, v in self.metrics.items()},
                },
                f,
                indent=2,
            )
        print(f"✅ Profiling results saved to: {output_path}")

    def print_summary(self):
        """Print formatted profiling summary."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        for name, stats in sorted(summary.items()):
            print(f"\n📊 {name}:")
            print(f"   Mean:  {stats['mean']*1000:.2f} ms")
            print(f"   Std:   {stats['std']*1000:.2f} ms")
            print(f"   Min:   {stats['min']*1000:.2f} ms")
            print(f"   Max:   {stats['max']*1000:.2f} ms")
            print(f"   Total: {stats['total']:.2f} s")
            print(f"   Count: {stats['count']}")

        print("\n" + "=" * 80)


@contextmanager
def profile_pytorch(
    output_dir: str = "profiling_results",
    use_cuda: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
):
    """
    Profile PyTorch operations with detailed GPU/CPU metrics.

    Usage:
        with profile_pytorch():
            # your training code here
            pass
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_path)),
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof

    # Save text report
    with open(output_path / "profile_report.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total" if use_cuda else "cpu_time_total", row_limit=50))

    print(f"\n✅ PyTorch profiling results saved to: {output_path}")
    print(f"   View in TensorBoard: tensorboard --logdir {output_path}")


@contextmanager
def profile_python(output_file: str = "profiling_results/python_profile.stats"):
    """
    Profile Python function calls using cProfile.

    Usage:
        with profile_python("my_profile.stats"):
            # your code here
            pass
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield profiler
    finally:
        profiler.disable()

        # Save stats
        profiler.dump_stats(str(output_path))

        # Print top functions
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)

        report = s.getvalue()
        print("\n" + "=" * 80)
        print("PYTHON PROFILING REPORT (Top 30 functions)")
        print("=" * 80)
        print(report)

        # Save text report
        with open(output_path.with_suffix(".txt"), "w") as f:
            f.write(report)

        print(f"\n✅ Python profiling saved to: {output_path}")


def profile_dataloader(dataloader, num_batches: int = 100) -> Dict[str, float]:
    """
    Profile DataLoader performance.

    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to profile

    Returns:
        Dictionary with timing statistics
    """
    print(f"\n📊 Profiling DataLoader for {num_batches} batches...")

    times = []
    start_total = time.perf_counter()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_start = time.perf_counter()
        # Simulate data usage (move to device if needed)
        if isinstance(batch, (tuple, list)):
            _ = [b.shape if hasattr(b, "shape") else b for b in batch]
        batch_time = time.perf_counter() - batch_start
        times.append(batch_time)

    total_time = time.perf_counter() - start_total

    import numpy as np

    stats = {
        "mean_batch_time": float(np.mean(times)),
        "std_batch_time": float(np.std(times)),
        "min_batch_time": float(np.min(times)),
        "max_batch_time": float(np.max(times)),
        "total_time": total_time,
        "throughput": num_batches / total_time,
    }

    print(f"\n   Mean batch time: {stats['mean_batch_time']*1000:.2f} ms")
    print(f"   Throughput: {stats['throughput']:.2f} batches/sec")
    print(f"   Total time: {stats['total_time']:.2f} s")

    return stats


class GPUMemoryProfiler:
    """Profile GPU memory usage during training."""

    def __init__(self):
        self.snapshots = []

    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self.snapshots.append(
                {"label": label, "allocated_gb": allocated, "reserved_gb": reserved, "time": time.time()}
            )

    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            print("No GPU memory snapshots taken")
            return

        print("\n" + "=" * 80)
        print("GPU MEMORY USAGE")
        print("=" * 80)

        for snap in self.snapshots:
            print(f"{snap['label']:30s} | Allocated: {snap['allocated_gb']:.2f} GB | Reserved: {snap['reserved_gb']:.2f} GB")

        print("=" * 80)

    def get_peak_memory(self) -> float:
        """Get peak allocated memory in GB."""
        if not self.snapshots:
            return 0.0
        return max(s["allocated_gb"] for s in self.snapshots)


def analyze_bottlenecks(profile_summary: Dict[str, Any], threshold_ms: float = 100.0):
    """
    Analyze profiling results and identify bottlenecks.

    Args:
        profile_summary: Summary from TrainingProfiler.get_summary()
        threshold_ms: Time threshold in ms to flag as bottleneck
    """
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    bottlenecks = []
    for name, stats in profile_summary.items():
        mean_ms = stats["mean"] * 1000
        if mean_ms > threshold_ms:
            bottlenecks.append((name, mean_ms, stats["total"]))

    if not bottlenecks:
        print(f"✅ No bottlenecks detected (threshold: {threshold_ms} ms)")
        return

    print(f"⚠️  Found {len(bottlenecks)} potential bottlenecks (>{threshold_ms} ms):\n")

    for name, mean_ms, total_s in sorted(bottlenecks, key=lambda x: x[1], reverse=True):
        print(f"   🔴 {name}")
        print(f"      Mean time: {mean_ms:.2f} ms")
        print(f"      Total time: {total_s:.2f} s")
        print()

    print("💡 Optimization suggestions:")
    print("   - Use DataLoader with num_workers > 0 for parallel data loading")
    print("   - Enable pin_memory=True for faster GPU transfers")
    print("   - Use mixed precision training (torch.cuda.amp)")
    print("   - Profile with PyTorch profiler for detailed GPU analysis")
    print("   - Consider gradient accumulation to reduce batch operations")
    print("=" * 80)
