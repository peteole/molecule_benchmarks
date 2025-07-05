#!/usr/bin/env python3
"""Test script to verify caching with different datasets."""

import sys
import time

sys.path.insert(0, "/Users/ole/Documents/software/molecule_benchmarks")

from molecule_benchmarks.benchmarker import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset


def test_different_datasets():
    """Test that different datasets use different cache keys."""
    print("Testing different datasets...")

    # Load dummy dataset
    dataset1 = SmilesDataset.load_dummy_dataset()

    # Create a modified dataset
    dataset2 = SmilesDataset(
        train_smiles=["CCO", "CCC", "CCN"], validation_smiles=["CCO", "CCC"]
    )

    # Initialize benchmarker with first dataset
    benchmarker1 = Benchmarker(dataset1, cache_dir="test_cache")
    hash1 = benchmarker1._generate_dataset_hash()

    # Initialize benchmarker with second dataset
    benchmarker2 = Benchmarker(dataset2, cache_dir="test_cache")
    hash2 = benchmarker2._generate_dataset_hash()

    print(f"Dataset 1 hash: {hash1}")
    print(f"Dataset 2 hash: {hash2}")

    # Verify they have different hashes
    assert hash1 != hash2, "Different datasets should have different cache keys"

    print("✓ Different datasets test passed!")


if __name__ == "__main__":
    test_different_datasets()
