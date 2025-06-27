from molecule_benchmarks import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset
from molecule_benchmarks.model import DummyMoleculeGenerationModel


def test_benchmarker():
    # Create a Benchmarker instance with some test SMILES
    ds = SmilesDataset.load_qm9_dataset(subset_size=10000)
    benchmarker = Benchmarker(ds)
    model = DummyMoleculeGenerationModel(ds.train_smiles[:5000])
    # Test the validity score computation
    scores = benchmarker.benchmark(model)
    print(scores)
    validity_scores = scores["validity"]
    assert validity_scores["valid_fraction"] >= 0.99, (
        f"Expected valid fraction of almost 100% but got {validity_scores['valid_fraction']}"
    )
    assert validity_scores["valid_and_unique_fraction"] <= 5000 / 10000, (
        f"Got {validity_scores['valid_and_unique_fraction']}"
    )
    assert 0.49 <= validity_scores["unique_fraction"] <= 0.5, (
        f"Expected 5000/10000, but got {validity_scores['unique_fraction']}"
    )
    assert scores["kl_score"] > 0.95, (
        f"Expected KL score to be high, got {scores['kl_score']}"
    )
    assert scores["fcd"]["fcd"] < 0.3, (
        f"Expected FCD score to be low, got {scores['fcd']['fcd']}"
    )
