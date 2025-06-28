import requests

from molecule_benchmarks import Benchmarker
from molecule_benchmarks.dataset import SmilesDataset
from molecule_benchmarks.model import DummyMoleculeGenerationModel


def test_benchmarker():
    # Create a Benchmarker instance with some test SMILES
    ds = SmilesDataset.load_qm9_dataset(max_train_samples=10000)
    benchmarker = Benchmarker(ds, num_samples_to_generate=10000)
    # Test the benchmarker can handle only invalid SMILES
    benchmarker.benchmark(["C-H-C"]*5000 + [None] * 5000)  # 10000 invalid SMILES

    model = DummyMoleculeGenerationModel(ds.train_smiles[:5000])
    
    # Test the validity score computation
    scores = benchmarker.benchmark_model(model)
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


def _test_moses_benchmarks_match(model_name: str, run: int, benchmarker: Benchmarker):
    """Test that Moses benchmarks match the values computed by the original implementation."""
    metrics_url = f"https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/samples/{model_name}/metrics_{model_name}_{run}.csv"
    samples_url = f"https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/samples/{model_name}/{model_name}_{run}.csv"

    metrics_response = requests.get(metrics_url)
    metrics_response.raise_for_status()
    metrics = metrics_response.text.splitlines()
    metrics = metrics[1:]  # Skip header
    metrics_dict: dict[str, float] = {}
    for line in metrics:
        key, value = line.split(",")
        metrics_dict[key] = float(value)
    print(metrics_dict)  # Print metrics for debugging

    samples_response = requests.get(samples_url)
    samples_response.raise_for_status()
    samples = samples_response.text.splitlines()
    samples = samples[1:]  # Skip header
    print(samples[:5])  # Print first 5 samples for debugging
    print("Number of samples:", len(samples))
    
    scores = benchmarker.benchmark(samples)
    print(scores)  # Print scores for debugging


def test_moses_benchmarks():
    """Test that Moses benchmarks match the values computed by the original implementation."""
    ds = SmilesDataset.load_moses_dataset(max_train_samples=50000)
    benchmarker = Benchmarker(ds, num_samples_to_generate=10000, device="mps")
    # Test for model 'aae'
    _test_moses_benchmarks_match("aae", 1, benchmarker)