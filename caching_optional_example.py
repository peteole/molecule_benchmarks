#!/usr/bin/env python3
"""Example showing how to use the Benchmarker with optional caching."""

from molecule_benchmarks.benchmarker import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset


def main():
    """Example of using Benchmarker with and without caching."""
    
    # Load a dummy dataset
    dataset = SmilesDataset.load_dummy_dataset()
    
    print("=== Example 1: Benchmarker with caching disabled ===")
    benchmarker_no_cache = Benchmarker(dataset, cache_dir=None)
    print(f"Cache directory: {benchmarker_no_cache.cache_dir}")
    print("No cache files will be created.\n")
    
    print("=== Example 2: Benchmarker with caching enabled (default) ===")
    benchmarker_with_cache = Benchmarker(dataset, cache_dir="cache_demo")
    print(f"Cache directory: {benchmarker_with_cache.cache_dir}")
    print("Cache files will be saved for faster subsequent runs.\n")
    
    print("=== Example 3: Benchmarker with default caching ===")
    benchmarker_default = Benchmarker(dataset)  # Uses default cache_dir="data"
    print(f"Cache directory: {benchmarker_default.cache_dir}")
    print("Uses default 'data' directory for caching.")


if __name__ == "__main__":
    main()
