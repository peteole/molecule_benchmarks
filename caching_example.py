#!/usr/bin/env python3
"""Example demonstrating the caching feature."""

from molecule_benchmarks.benchmarker import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset


def main():
    # Load a dataset
    dataset = SmilesDataset.load_dummy_dataset()

    # First initialization - will compute and cache validation statistics
    print("First initialization (computing and caching)...")
    benchmarker1 = Benchmarker(dataset, cache_dir="cache_demo")

    # Second initialization - will load from cache
    print("\nSecond initialization (loading from cache)...")
    benchmarker2 = Benchmarker(dataset, cache_dir="cache_demo")

    print("\nBoth benchmarkers are ready to use!")
    print(f"Validation set size: {len(dataset.get_validation_smiles())}")
    print(f"Cache directory: cache_demo/")


if __name__ == "__main__":
    main()
