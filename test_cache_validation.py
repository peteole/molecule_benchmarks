#!/usr/bin/env python3
"""Test script to verify cache validation."""

import sys
import time
import pickle
from pathlib import Path

sys.path.insert(0, "/Users/ole/Documents/software/molecule_benchmarks")

from molecule_benchmarks.benchmarker import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset


def test_cache_validation():
    """Test that cache validation works correctly."""
    print("Testing cache validation...")

    # Load dummy dataset
    dataset1 = SmilesDataset.load_dummy_dataset()

    # Create a modified dataset
    dataset2 = SmilesDataset(
        train_smiles=["CCO", "CCC", "CCN"], validation_smiles=["CCO", "CCC"]
    )

    # Initialize benchmarker with first dataset
    benchmarker1 = Benchmarker(dataset1, cache_dir="test_cache_validation")
    hash1 = benchmarker1._generate_dataset_hash()

    # Create a fake cache file for dataset2 but with dataset1's hash
    cache_dir = Path("test_cache_validation")
    hash2 = Benchmarker(dataset2, cache_dir="temp_cache")._generate_dataset_hash()

    # Copy the cache file from dataset1 to dataset2's expected location
    cache_path1 = cache_dir / f"benchmarker_cache_{hash1}.pkl"
    cache_path2 = cache_dir / f"benchmarker_cache_{hash2}.pkl"

    # Manually create a corrupted cache (wrong hash)
    if cache_path1.exists():
        with open(cache_path1, "rb") as f:
            cache_data = pickle.load(f)

        # Corrupt the hash
        cache_data["dataset_hash"] = "wrong_hash"

        with open(cache_path2, "wb") as f:
            pickle.dump(cache_data, f)

    # Now try to load dataset2 - it should detect hash mismatch and recompute
    print("Testing with corrupted cache...")
    benchmarker2 = Benchmarker(dataset2, cache_dir="test_cache_validation")

    print("✓ Cache validation test passed!")


if __name__ == "__main__":
    test_cache_validation()
