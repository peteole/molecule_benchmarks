from molecule_benchmarks.dataset import SmilesDataset


def test_guacamol():
    # Test loading the Guacamole  dataset
    dataset = SmilesDataset.load_guacamol_dataset(0.001)
    assert len(dataset.train_smiles) > 10, "Guacamole dataset has no training SMILES"
    assert len(dataset.validation_smiles) > 10, (
        "Guacamole dataset has no validation SMILES"
    )


def test_moses():
    # Test loading the Moses dataset
    dataset = SmilesDataset.load_moses_dataset(0.001)
    assert len(dataset.train_smiles) > 10, "Moses dataset has no training SMILES"
    assert len(dataset.validation_smiles) > 10, "Moses dataset has no validation SMILES"


def test_dummy():
    # Test loading the dummy dataset
    dataset = SmilesDataset.load_dummy_dataset()
    assert len(dataset.train_smiles) > 0, "Dummy dataset has no training SMILES"
    assert len(dataset.validation_smiles) > 0, "Dummy dataset has no validation SMILES"


def test_qm9():
    # Test loading the QM9 dataset
    dataset = SmilesDataset.load_qm9_dataset(subset_size=1000)
    assert len(dataset.train_smiles) > 10, "QM9 dataset has no training SMILES"
    assert len(dataset.validation_smiles) > 10, "QM9 dataset has no validation SMILES"
