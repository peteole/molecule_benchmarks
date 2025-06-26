import math
from typing import TypedDict
from fcd import get_fcd  # type: ignore
from tqdm import tqdm  # type: ignore
from rdkit import Chem
from molecule_benchmarks.dataset import SmilesDataset, canonicalize_smiles_list
from molecule_benchmarks.model import MoleculeGenerationModel


class ValidityBenchmarkResults(TypedDict):
    num_molecules_generated: int
    valid_fraction: float
    valid_and_unique_fraction: float
    unique_fraction: float
    unique_and_novel_fraction: float
    valid_and_unique_and_novel_fraction: float


class FCDBenchmarkResults(TypedDict):
    """Fréchet ChemNet Distance (FCD) benchmark results."""

    fcd: float
    "The FCD score for the generated molecules."
    fcd_valid: float
    "The FCD score for the valid generated molecules."
    fcd_normalized: float
    "The normalized FCD score for the generated molecules, calculated as exp(-0.2 * fcd)."
    fcd_valid_normalized: float
    "The normalized FCD score for the valid generated molecules, calculated as exp(-0.2 * fcd_valid)."


class BenchmarkResults(TypedDict):
    """Combined benchmark results."""

    validity: ValidityBenchmarkResults
    fcd: FCDBenchmarkResults


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
        return False


class Benchmarker:
    """Benchmarker for evaluating molecule generation models."""

    def __init__(
        self,
        dataset: SmilesDataset,
        num_samples_to_generate: int = 10000,
        device: str = "cpu",
    ) -> None:
        self.dataset = dataset
        self.num_samples_to_generate = num_samples_to_generate
        self.device = device

    def benchmark(self, model: MoleculeGenerationModel) -> BenchmarkResults:
        """Run the benchmarks on the generated SMILES."""

        generated_smiles = model.generate_molecules(self.num_samples_to_generate)
        if not generated_smiles:
            raise ValueError("No generated SMILES provided for benchmarking.")
        generated_smiles = canonicalize_smiles_list(generated_smiles)
        validity_results = self._compute_validity_scores(generated_smiles)
        fcd_results = self._compute_fcd_scores(generated_smiles)

        return {"validity": validity_results, "fcd": fcd_results}

    def _compute_validity_scores(
        self, generated_smiles: list[str | None]
    ) -> ValidityBenchmarkResults:
        valid_and_unique: set[str] = set()
        valid: list[str] = []
        unique: set[str] = set()
        existing = set(self.dataset.train_smiles)

        for smiles in tqdm(
            generated_smiles, desc="Generated molecules validity check progress"
        ):
            if smiles is not None:
                unique.add(smiles)

                if is_valid_smiles(smiles):
                    valid_and_unique.add(smiles)
                    valid.append(smiles)
        valid_and_unique_fraction = (
            len(valid_and_unique) / len(generated_smiles) if generated_smiles else 0.0
        )
        valid_fraction = len(valid) / len(generated_smiles) if generated_smiles else 0.0
        unique_fraction = (
            len(unique) / len(generated_smiles) if generated_smiles else 0.0
        )
        unique_and_novel_fraction = (
            len(unique - existing) / len(generated_smiles) if generated_smiles else 0.0
        )
        valid_and_unique_and_novel_fraction = (
            len(valid_and_unique - existing) / len(generated_smiles)
            if generated_smiles
            else 0.0
        )
        return {
            "num_molecules_generated": len(generated_smiles),
            "valid_fraction": valid_fraction,
            "valid_and_unique_fraction": valid_and_unique_fraction,
            "unique_fraction": unique_fraction,
            "unique_and_novel_fraction": unique_and_novel_fraction,
            "valid_and_unique_and_novel_fraction": valid_and_unique_and_novel_fraction,
        }

    def _compute_fcd_scores(
        self, generated_smiles: list[str | None]
    ) -> FCDBenchmarkResults:
        """Compute the Fréchet ChemNet Distance (FCD) scores for the generated SMILES. Removes any None-type smiles."""

        valid_generated_smiles = [
            smiles
            for smiles in generated_smiles
            if smiles is not None and is_valid_smiles(smiles)
        ]

        fcd_score = get_fcd(
            [s for s in generated_smiles if s is not None],
            self.dataset.validation_smiles,
            device=self.device,
        )
        fcd_valid_score = get_fcd(
            valid_generated_smiles, self.dataset.validation_smiles, device=self.device
        )
        return {
            "fcd": fcd_score,
            "fcd_valid": fcd_valid_score,
            "fcd_normalized": math.exp(-0.2 * fcd_score),
            "fcd_valid_normalized": math.exp(-0.2 * fcd_valid_score),
        }


