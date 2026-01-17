#!/usr/bin/env python
"""
Profiling script for training performance analysis.

This script adds comprehensive profiling to the training loop to identify bottlenecks
and optimize performance.

Usage:
    # Profile with default settings
    python scripts/profile_training.py

    # Profile specific config
    python scripts/profile_training.py --config-name=quick_test

    # Profile with PyTorch profiler
    python scripts/profile_training.py --use-pytorch-profiler

    # Profile with Python cProfile
    python scripts/profile_training.py --use-python-profiler
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from energy_dissagregation_mlops.model import Model
from energy_dissagregation_mlops.data import MyDataset
from energy_dissagregation_mlops.profiling import (
    TrainingProfiler,
    profile_pytorch,
    profile_python,
    profile_dataloader,
    GPUMemoryProfiler,
    analyze_bottlenecks,
)


def train_with_profiling(
    preprocessed_folder: str = "data/processed_fast",
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 3,
    num_workers: int = 2,
    device: str | None = None,
    use_pytorch_profiler: bool = False,
    use_python_profiler: bool = False,
    profile_dataloader_only: bool = False,
) -> None:
    """
    Train model with comprehensive profiling.

    Args:
        preprocessed_folder: Path to preprocessed data
        batch_size: Batch size
        lr: Learning rate
        epochs: Number of epochs
        num_workers: DataLoader workers
        device: Device to use (cuda/cpu/auto)
        use_pytorch_profiler: Enable PyTorch profiler
        use_python_profiler: Enable Python cProfile
        profile_dataloader_only: Only profile DataLoader performance
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize profilers
    profiler = TrainingProfiler(output_dir="profiling_results")
    gpu_mem = GPUMemoryProfiler()

    with profiler.profile_section("data_loading"):
        dataset = MyDataset(
            data_path=Path("data/raw/ukdale.h5"),
            preprocessed_folder=Path(preprocessed_folder),
            window_size=256,
            stride=256,
        )

        n = len(dataset)
        n_train = int(0.9 * n)
        n_val = n - n_train
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    print(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val")

    # Profile DataLoader if requested
    if profile_dataloader_only:
        print("\n📊 Profiling DataLoader performance...")
        train_stats = profile_dataloader(train_loader, num_batches=100)
        profiler.save_results("dataloader_profile.json")
        return

    with profiler.profile_section("model_initialization"):
        model = Model(window_size=1024).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

    gpu_mem.snapshot("After model initialization")

    best_val = float("inf")
    ckpt_path = Path("models")
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Wrap training in profilers if requested
    profiling_context = None
    if use_pytorch_profiler:
        profiling_context = profile_pytorch(use_cuda=(device == "cuda"))
    elif use_python_profiler:
        profiling_context = profile_python()

    if profiling_context:
        profiling_context.__enter__()

    try:
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")

            # ---- TRAIN ----
            model.train()
            train_loss = 0.0

            with profiler.profile_section(f"epoch_{epoch}_train"):
                for batch_idx, (x, y) in enumerate(train_loader):
                    with profiler.profile_section("data_transfer"):
                        x, y = x.to(device), y.to(device)

                    with profiler.profile_section("forward_pass"):
                        y_hat = model(x)

                    with profiler.profile_section("loss_computation"):
                        loss = criterion(y_hat, y)

                    with profiler.profile_section("backward_pass"):
                        optimizer.zero_grad()
                        loss.backward()

                    with profiler.profile_section("optimizer_step"):
                        optimizer.step()

                    train_loss += loss.item()

                    # Memory snapshot every 50 batches
                    if batch_idx % 50 == 0:
                        gpu_mem.snapshot(f"Epoch {epoch}, Batch {batch_idx}")

                    # Profile first 10 batches in detail
                    if batch_idx < 10:
                        profiler.record_metric("batch_loss", loss.item())

            avg_train_loss = train_loss / len(train_loader)
            profiler.record_metric("train_loss_per_epoch", avg_train_loss)

            # ---- VALIDATION ----
            model.eval()
            val_loss = 0.0

            with profiler.profile_section(f"epoch_{epoch}_validation"):
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                        val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            profiler.record_metric("val_loss_per_epoch", avg_val_loss)

            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                with profiler.profile_section("model_save"):
                    torch.save(model.state_dict(), ckpt_path / "best.pt")
                print(f"✅ New best model saved (val_loss={best_val:.4f})")

            gpu_mem.snapshot(f"End of epoch {epoch}")

    finally:
        if profiling_context:
            profiling_context.__exit__(None, None, None)

    # Print and save results
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)

    profiler.print_summary()
    gpu_mem.print_summary()

    # Analyze bottlenecks
    summary = profiler.get_summary()
    analyze_bottlenecks(summary, threshold_ms=50.0)

    # Save results
    profiler.save_results("training_profile.json")

    print(f"\n✅ Training completed!")
    print(f"   Best validation loss: {best_val:.4f}")
    print(f"   Peak GPU memory: {gpu_mem.get_peak_memory():.2f} GB")
    print(f"\n📊 Profiling results saved to: profiling_results/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("--preprocessed-folder", type=str, default="data/processed_fast", help="Preprocessed data folder")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--use-pytorch-profiler", action="store_true", help="Use PyTorch profiler")
    parser.add_argument("--use-python-profiler", action="store_true", help="Use Python cProfile")
    parser.add_argument("--profile-dataloader-only", action="store_true", help="Only profile DataLoader")

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_with_profiling(
        preprocessed_folder=args.preprocessed_folder,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        device=device,
        use_pytorch_profiler=args.use_pytorch_profiler,
        use_python_profiler=args.use_python_profiler,
        profile_dataloader_only=args.profile_dataloader_only,
    )


if __name__ == "__main__":
    main()
