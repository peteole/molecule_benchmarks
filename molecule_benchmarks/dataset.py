import csv
import itertools
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, TypeVar

import requests
from rdkit import Chem
from tqdm import tqdm  # type: ignore


class SmilesDataset:
    @classmethod
    def load_qm9_dataset(cls, subset_size: Optional[int] = None):
        """Load the QM9 dataset."""
        ds_url = (
            "https://huggingface.co/datasets/n0w0f/qm9-csv/resolve/main/qm9_dataset.csv"
        )
        response = requests.get(ds_url)
        response.raise_for_status()
        smiles = []

        reader = csv.DictReader(response.text.splitlines())
        if subset_size is not None:
            reader = itertools.islice(reader, subset_size)

        for row in reader:
            smiles.append(row["smiles"])  # Assuming the column name is "smiles"
        # smiles = canonicalize_smiles_list(smiles)
        random.seed(42)  # For reproducibility
        random.shuffle(smiles)
        num_train = int(0.8 * len(smiles))
        train_smiles = smiles[:num_train]
        validation_smiles = smiles[num_train:]
        return cls(train_smiles=train_smiles, validation_smiles=validation_smiles)

    @classmethod
    def load_guacamol_dataset(cls, fraction: float = 1.0):
        """Load the Guacamole dataset."""
        train_ds_url = "https://ndownloader.figshare.com/files/13612760"
        validation_ds_url = "https://ndownloader.figshare.com/files/13612766"
        # download the dataset into memory
        response = requests.get(train_ds_url)
        response.raise_for_status()
        train_smiles = response.text.splitlines()
        random.seed(42)  # For reproducibility
        if fraction < 1.0:
            train_smiles = random.sample(
                train_smiles, int(len(train_smiles) * fraction)
            )
        response = requests.get(validation_ds_url)
        response.raise_for_status()
        validation_smiles = response.text.splitlines()
        if fraction < 1.0:
            validation_smiles = random.sample(
                validation_smiles, int(len(validation_smiles) * fraction)
            )
        return cls(train_smiles=train_smiles, validation_smiles=validation_smiles)

    @classmethod
    def load_moses_dataset(cls, fraction: float = 1.0):
        """Load the Moses dataset."""

        def download_smiles(split: str) -> list[str]:
            """Download SMILES from a given URL split."""
            url = f"https://media.githubusercontent.com/media/molecularsets/moses/master/data/{split}.csv"
            response = requests.get(url)
            response.raise_for_status()
            csv_file = response.text.splitlines()
            if fraction < 1.0:
                # Sample a fraction of the dataset
                csv_file = [csv_file[0]] + random.sample(
                    csv_file[1:], int(len(csv_file) * fraction)
                )
            reader = csv.DictReader(csv_file)
            smiles = [row["SMILES"] for row in reader]
            return smiles

        train_smiles = download_smiles("train")
        validation_smiles = download_smiles("test")
        return cls(train_smiles=train_smiles, validation_smiles=validation_smiles)

    @classmethod
    def load_dummy_dataset(cls):
        """Load a dummy dataset for testing purposes."""
        train_smiles = ["C1=CC=CC=C1", "C1=CC=CN=C1", "C1=CC=CO=C1"]
        validation_smiles = ["C1=CC=CC=C1", "C1=CC=CN=C1"]
        return cls(train_smiles=train_smiles, validation_smiles=validation_smiles)

    def __init__(
        self,
        train_smiles: list[str] | Path | str,
        validation_smiles: list[str] | Path | str,
    ) -> None:
        self.train_smiles = load_smiles(train_smiles)
        self.validation_smiles = load_smiles(validation_smiles)

    def get_train_smiles(self) -> list[str]:
        """Get the training SMILES."""
        return self.train_smiles

    def get_validation_smiles(self) -> list[str]:
        """Get the validation SMILES."""
        return self.validation_smiles

    def get_train_molecules(self) -> list[Chem.Mol | None]:
        """Get the training molecules."""
        return [Chem.MolFromSmiles(s) for s in self.train_smiles]

    def get_validation_molecules(self) -> list[Chem.Mol | None]:
        """Get the validation molecules."""
        return [Chem.MolFromSmiles(s) for s in self.validation_smiles]


def load_smiles(smiles: list[str] | str | Path) -> list[str]:
    """Load SMILES from a file or a list. Canonicalizes the SMILES strings."""
    if isinstance(smiles, str):
        smiles = Path(smiles)
    if isinstance(smiles, Path):
        with smiles.open("r") as f:
            smiles = f.readlines()
    smiles = [s.strip() for s in smiles if s.strip()]
    canonicalized_smiles = canonicalize_smiles_list(smiles)
    # Filter out None values that couldn't be canonicalized
    return [s for s in canonicalized_smiles if s is not None]


T = TypeVar("T", str, Optional[str])


def _canonicalize_single_smiles(smiles: Optional[str]) -> Optional[str]:
    """Helper function to canonicalize a single SMILES string for multiprocessing."""
    if smiles is not None:
        try:
            return Chem.CanonSmiles(smiles)
        except Exception:
            return smiles
    return None


def canonicalize_smiles_list(smiles: list[T], n_jobs: Optional[int] = None) -> list[T]:
    """Canonicalize a list of SMILES strings using multiprocessing with progress tracking."""

    if n_jobs is None:
        n_jobs = mp.cpu_count()

    # Handle empty list
    if not smiles:
        return []

    # For small lists, use serial processing
    if len(smiles) < 100:
        return [
            _canonicalize_single_smiles(s)  # type: ignore
            for s in tqdm(smiles, desc="Canonicalizing SMILES")
        ]

    # Use multiprocessing for larger lists
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(
            tqdm(
                executor.map(_canonicalize_single_smiles, smiles),
                total=len(smiles),
                desc="Canonicalizing SMILES",
            )
        )

    return results
