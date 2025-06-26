from molecule_benchmarks import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset

def test_benchmarker():
    # Create a Benchmarker instance with some test SMILES
    ds  = SmilesDataset.load_dummy_dataset()
    benchmarker = Benchmarker(ds)

    # Test the validity score computation
    validity_scores = benchmarker._compute_validity_scores([
        "C1=CC=CC=C1", "C-H-O", "C1=CC=CC=C1",  # Valid SMILES
    ])
    assert validity_scores["valid_fraction"] == 2/3, f"Expected 2/3, but got {validity_scores['valid_fraction']}"
    assert validity_scores["valid_and_unique_fraction"] == 1/3, f"Expected 1/3, but got {validity_scores['valid_and_unique_fraction']}"
    assert validity_scores["unique_fraction"] == 2/3, f"Expected 2/3, but got {validity_scores['unique_fraction']}"


def test_fcd():
    # Test loading the Guacamole  dataset and computing FCD scores
    dataset  = SmilesDataset.load_guacamole_dataset()
    benchmarker = Benchmarker(dataset)
    assert isinstance(benchmarker, Benchmarker), "Failed to load Guacamole benchmarker"
    assert len(benchmarker.dataset.train_smiles) > 10, "Guacamole benchmarker has no training SMILES"
    dataset.validation_smiles = dataset.validation_smiles[:100]  # Use a subset for testing
    fcd_to_self = benchmarker._compute_fcd_scores(benchmarker.dataset.validation_smiles)
    assert fcd_to_self["fcd"] <= 1e-4, "FCD score to self should be 0.0"