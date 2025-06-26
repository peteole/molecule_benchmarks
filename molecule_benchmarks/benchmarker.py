import math
from typing import TypedDict
from fcd import get_fcd  # type: ignore
from tqdm import tqdm  # type: ignore
from rdkit import Chem
from molecule_benchmarks.dataset import SmilesDataset, canonicalize_smiles_list
from molecule_benchmarks.model import MoleculeGenerationModel
from molecule_benchmarks.utils import calculate_internal_pairwise_similarities, calculate_pc_descriptors, continuous_kldiv, discrete_kldiv
import numpy as np


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
    kl_score: float
    "KL score from guacamol (https://arxiv.org/pdf/1811.09621). In [0,1]"


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
        kl_score = self._compute_kl_score(generated_smiles)
        validity_results = self._compute_validity_scores(generated_smiles)
        fcd_results = self._compute_fcd_scores(generated_smiles)

        return {"validity": validity_results, "fcd": fcd_results, "kl_score": kl_score}

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
        print("Computing FCD scores for the generated SMILES...")
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

    def _compute_kl_score(self, generated_smiles: list[str | None]) -> float:
        """Compute the KL divergence score for the generated SMILES."""
        print("Computing KL divergence score for the generated SMILES...")
        pc_descriptor_subset = [
            "BertzCT",
            "MolLogP",
            "MolWt",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "NumAliphaticRings",
            "NumAromaticRings",
        ]
        generated_smiles_valid = [s for s in generated_smiles if s is not None]
        unique_molecules = set(generated_smiles_valid)

        d_sampled = calculate_pc_descriptors(
            generated_smiles_valid, pc_descriptor_subset
        )
        d_chembl = calculate_pc_descriptors(
            self.dataset.get_train_smiles(), pc_descriptor_subset
        )

        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i in range(4):
            kldiv = continuous_kldiv(
                X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i]
            )
            kldivs[pc_descriptor_subset[i]] = kldiv

        # ... and for the int valued ones.
        for i in range(4, 9):
            kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[pc_descriptor_subset[i]] = kldiv

        # pairwise similarity

        chembl_sim = calculate_internal_pairwise_similarities(
            self.dataset.get_train_smiles()
        )
        chembl_sim = chembl_sim.max(axis=1)

        sampled_sim = calculate_internal_pairwise_similarities(unique_molecules)
        sampled_sim = sampled_sim.max(axis=1)

        kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
        kldivs["internal_similarity"] = kldiv_int_int

        # for some reason, this runs into problems when both sets are identical.
        # cross_set_sim = calculate_pairwise_similarities(self.training_set_molecules, unique_molecules)
        # cross_set_sim = cross_set_sim.max(axis=1)
        #
        # kldiv_ext = discrete_kldiv(chembl_sim, cross_set_sim)
        # kldivs['external_similarity'] = kldiv_ext
        # kldiv_sum += kldiv_ext

        #metadata = {"number_samples": self.number_samples, "kl_divs": kldivs}

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)
        print("KL divergence score:", score)
        return score
