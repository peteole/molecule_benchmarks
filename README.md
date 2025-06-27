# Molecule Benchmarks

[![PyPI version](https://badge.fury.io/py/molecule-benchmarks.svg)](https://badge.fury.io/py/molecule-benchmarks)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive benchmark suite for evaluating generative models for molecules. This package provides standardized metrics and evaluation protocols for assessing the quality of molecular generation models in drug discovery and cheminformatics.

## Features

- **Comprehensive Metrics**: Validity, uniqueness, novelty, diversity, and similarity metrics
- **Standard Benchmarks**: Implements metrics from Moses, GuacaMol, and FCD papers
- **Easy Integration**: Simple interface for integrating with any generative model
- **Multiple Datasets**: Built-in support for QM9, Moses, and GuacaMol datasets
- **Efficient Computation**: Optimized for large-scale evaluation with multiprocessing support

## Installation

```bash
pip install molecule-benchmarks
```

## Quick Start

### 1. Implement Your Model

To use the benchmark suite, implement the `MoleculeGenerationModel` protocol:

```python
from molecule_benchmarks.model import MoleculeGenerationModel

class MyGenerativeModel(MoleculeGenerationModel):
    def __init__(self, model_path):
        # Initialize your model here
        self.model = load_model(model_path)
    
    def generate_molecule_batch(self) -> list[str | None]:
        """Generate a batch of molecules as SMILES strings.
        
        Returns:
            List of SMILES strings. Return None for invalid molecules.
        """
        # Your generation logic here
        batch = self.model.generate(batch_size=100)
        return [self.convert_to_smiles(mol) for mol in batch]
```

### 2. Run Benchmarks

```python
from molecule_benchmarks import Benchmarker, SmilesDataset

# Load a dataset
dataset = SmilesDataset.load_qm9_dataset(subset_size=10000)

# Initialize benchmarker
benchmarker = Benchmarker(
    dataset=dataset,
    num_samples_to_generate=10000,
    device="cpu"  # or "cuda" for GPU
)

# Initialize your model
model = MyGenerativeModel("path/to/model")

# Run benchmarks
results = benchmarker.benchmark(model)
print(results)
```

### 3. Analyze Results

The benchmark returns comprehensive metrics:

```python
# Validity metrics
print(f"Valid molecules: {results['validity']['valid_fraction']:.3f}")
print(f"Valid & unique: {results['validity']['valid_and_unique_fraction']:.3f}")
print(f"Valid & unique & novel: {results['validity']['valid_and_unique_and_novel_fraction']:.3f}")

# Diversity and similarity metrics
print(f"Internal diversity: {results['moses']['IntDiv']:.3f}")
print(f"SNN score: {results['moses']['snn_score']:.3f}")

# Chemical property distribution similarity
print(f"KL divergence score: {results['kl_score']:.3f}")

# Fréchet ChemNet Distance
print(f"FCD score: {results['fcd']['fcd']:.3f}")
```

## Complete Example

Here's a complete example using the built-in dummy model:

```python
from molecule_benchmarks import Benchmarker, SmilesDataset
from molecule_benchmarks.model import DummyMoleculeGenerationModel

# Load dataset
print("Loading dataset...")
dataset = SmilesDataset.load_qm9_dataset(subset_size=1000)

# Create benchmarker
benchmarker = Benchmarker(
    dataset=dataset,
    num_samples_to_generate=100,
    device="cpu"
)

# Create a dummy model (replace with your model)
model = DummyMoleculeGenerationModel([
    "CCO",           # Ethanol
    "CC(=O)O",       # Acetic acid
    "c1ccccc1",      # Benzene
    "CC(C)O",        # Isopropanol
    "CCN",           # Ethylamine
    None,            # Invalid molecule
])

# Run benchmarks
print("Running benchmarks...")
results = benchmarker.benchmark(model)

# Print results
print("\n=== Validity Metrics ===")
print(f"Valid molecules: {results['validity']['valid_fraction']:.3f}")
print(f"Unique molecules: {results['validity']['unique_fraction']:.3f}")
print(f"Valid & unique: {results['validity']['valid_and_unique_fraction']:.3f}")
print(f"Novel molecules: {results['validity']['valid_and_unique_and_novel_fraction']:.3f}")

print("\n=== Moses Metrics ===")
print(f"Passing Moses filters: {results['moses']['fraction_passing_moses_filters']:.3f}")
print(f"SNN score: {results['moses']['snn_score']:.3f}")
print(f"Internal diversity (p=1): {results['moses']['IntDiv']:.3f}")
print(f"Internal diversity (p=2): {results['moses']['IntDiv2']:.3f}")

print("\n=== Distribution Metrics ===")
print(f"KL divergence score: {results['kl_score']:.3f}")
print(f"FCD score: {results['fcd']['fcd']:.3f}")
print(f"FCD (valid only): {results['fcd']['fcd_valid']:.3f}")
```

## Supported Datasets

The package includes several built-in datasets:

```python
from molecule_benchmarks import SmilesDataset

# QM9 dataset (small molecules)
dataset = SmilesDataset.load_qm9_dataset(subset_size=10000)

# Moses dataset (larger, drug-like molecules)
dataset = SmilesDataset.load_moses_dataset(fraction=0.1)

# GuacaMol dataset
dataset = SmilesDataset.load_guacamol_dataset(fraction=0.1)

# Custom dataset from files
dataset = SmilesDataset(
    train_smiles="path/to/train.txt",
    validation_smiles="path/to/valid.txt"
)
```

## Metrics Explained

### Validity Metrics

- **Valid fraction**: Percentage of generated molecules that are chemically valid
- **Unique fraction**: Percentage of generated molecules that are unique
- **Novel fraction**: Percentage of generated molecules not seen in training data

### Moses Metrics

Based on the [Moses paper](https://arxiv.org/abs/1811.12823):

- **SNN score**: Similarity to nearest neighbor in training set
- **Internal diversity**: Average pairwise Tanimoto distance within generated set
- **Scaffold similarity**: Similarity of molecular scaffolds to training set
- **Fragment similarity**: Similarity of molecular fragments to training set

### Distribution Metrics

- **KL divergence score**: Measures similarity of molecular property distributions
- **FCD score**: Fréchet ChemNet Distance, measures distribution similarity in learned feature space

## Advanced Usage

### Custom Evaluation

```python
# Custom number of samples and device
benchmarker = Benchmarker(
    dataset=dataset,
    num_samples_to_generate=50000,
    device="cuda"  # Use GPU for faster computation
)

# Run specific metric computations
results = benchmarker.benchmark(model)
validity_scores = benchmarker._compute_validity_scores(generated_smiles)
fcd_scores = benchmarker._compute_fcd_scores(generated_smiles)
```

### Batch Processing

```python
class BatchedModel(MoleculeGenerationModel):
    def generate_molecule_batch(self) -> list[str | None]:
        # Generate larger batches for efficiency
        return self.model.sample(batch_size=1000)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This benchmark suite implements and builds upon metrics from several important papers:

- [Moses: A Benchmarking Platform for Molecular Generation Models](https://arxiv.org/abs/1811.12823)
- [GuacaMol: Benchmarking Models for De Novo Molecular Design](https://arxiv.org/abs/1811.09621)
- [Fréchet ChemNet Distance: A Metric for Generative Models for Molecules](https://arxiv.org/abs/1803.09518)
