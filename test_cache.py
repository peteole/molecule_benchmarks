#!/usr/bin/env python3
"""Test script to verify caching functionality."""

import sys
import time

sys.path.insert(0, "/Users/ole/Documents/software/molecule_benchmarks")

from molecule_benchmarks.benchmarker import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset


def test_caching():
    """Test that caching works correctly."""
    print("Testing benchmarker caching...")

    # Load a dummy dataset
    dataset = SmilesDataset.load_dummy_dataset()

    # First initialization (should compute and cache)
    start_time = time.time()
    benchmarker1 = Benchmarker(dataset, cache_dir="test_cache")
    first_time = time.time() - start_time
    print(f"First initialization took {first_time:.2f} seconds")

    # Second initialization (should load from cache)
    start_time = time.time()
    benchmarker2 = Benchmarker(dataset, cache_dir="test_cache")
    second_time = time.time() - start_time
    print(f"Second initialization took {second_time:.2f} seconds")

    # Verify the cached values are the same
    assert benchmarker1.val_mu.shape == benchmarker2.val_mu.shape
    assert benchmarker1.val_sigma.shape == benchmarker2.val_sigma.shape
    assert len(benchmarker1.val_fingerprints_morgan) == len(
        benchmarker2.val_fingerprints_morgan
    )
    assert len(benchmarker1.val_scaffolds) == len(benchmarker2.val_scaffolds)
    assert len(benchmarker1.val_fragments) == len(benchmarker2.val_fragments)

    print("✓ Caching test passed!")


if __name__ == "__main__":
    test_caching()
