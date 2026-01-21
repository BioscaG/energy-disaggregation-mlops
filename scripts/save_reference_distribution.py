#!/usr/bin/env python3
"""
Save reference distribution from training data.

This script loads preprocessed training data and extracts the reference
distribution (baseline) to be used for drift detection in production.

The reference distribution represents the normal expected distribution
of model outputs during training.
"""

import os
from pathlib import Path

import numpy as np


def load_preprocessed_data(data_dir="data/preprocessed"):
    """Load all preprocessed data chunks."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all .npz files
    chunk_files = sorted(data_dir.glob("chunk_*.npz"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {data_dir}")

    print(f"Found {len(chunk_files)} data chunks")

    all_outputs = []

    for chunk_file in chunk_files:
        print(f"Loading {chunk_file.name}...", end=" ")
        data = np.load(chunk_file)

        # Print available keys
        print(f"Keys: {list(data.keys())}")

        # Extract outputs (usually the first or 'y' key)
        if "y" in data:
            outputs = data["y"]
        elif "X" in data:
            # Sometimes data is split into X and y
            outputs = data["y"] if "y" in data else data["X"]
        else:
            # Try first array
            outputs = data[list(data.keys())[0]]

        # Flatten if needed
        if len(outputs.shape) > 1:
            outputs = outputs.flatten()

        all_outputs.append(outputs)
        print(f"  Shape: {outputs.shape}, Range: [{outputs.min():.2f}, {outputs.max():.2f}]")

    # Combine all chunks
    reference_dist = np.concatenate(all_outputs, axis=0)

    print(f"\n✅ Reference distribution loaded")
    print(f"   Total samples: {len(reference_dist)}")
    print(f"   Mean: {reference_dist.mean():.4f}")
    print(f"   Std: {reference_dist.std():.4f}")
    print(f"   Min: {reference_dist.min():.4f}")
    print(f"   Max: {reference_dist.max():.4f}")

    return reference_dist


def save_reference_distribution(output_path="models/reference_distribution.npy"):
    """Main function to save reference distribution."""
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING REFERENCE DISTRIBUTION FOR DRIFT DETECTION")
    print("=" * 60 + "\n")

    # Load training data
    print("Step 1: Loading training data...")
    reference_dist = load_preprocessed_data()

    # Save to file
    print(f"\nStep 2: Saving to {output_path}...")
    np.save(output_path, reference_dist)

    # Verify
    loaded = np.load(output_path)
    print(f"✅ Saved successfully")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   Samples: {len(loaded)}")

    print("\n" + "=" * 60)

    return reference_dist


if __name__ == "__main__":
    save_reference_distribution()
