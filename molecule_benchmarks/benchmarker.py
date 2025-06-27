import math
from typing import TypedDict

import numpy as np
from fcd import get_fcd  # type: ignore
from rdkit import Chem
from tqdm import tqdm  # type: ignore

from molecule_benchmarks.dataset import SmilesDataset, canonicalize_smiles_list
from molecule_benchmarks.model import MoleculeGenerationModel
from molecule_benchmarks.moses_metrics import (
    average_agg_tanimoto,
    compute_fragments,
    compute_scaffolds,
    cos_similarity,
    fingerprints,
    internal_diversity,
    mapper,
    mol_passes_filters,
)
from molecule_benchmarks.utils import (
    calculate_internal_pairwise_similarities,
    calculate_pc_descriptors,
    continuous_kldiv,
    discrete_kldiv,
)


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


class MosesBenchmarkResults(TypedDict):
    """Moses benchmark results (see https://arxiv.org/abs/1811.12823)."""

    fraction_passing_moses_filters: float
    "Fraction of generated SMILES that pass the Moses filters (https://arxiv.org/abs/1811.12823)."
    snn_score: float
    "Similarity to a nearest neighbor (SNN) score from Moses (https://arxiv.org/abs/1811.12823). In [0,1], higher is better."
    IntDiv: float
    "Internal diversity score from Moses with p=1"
    IntDiv2: float
    "Internal diversity score from Moses with p=2"
    scaffolds_similarity: float
    "Scaffolds similarity metric from Moses. In [0,1], higher is better."
    fragment_similarity: float
    "Fragment similarity metric from Moses. In [0,1], higher is better."


class BenchmarkResults(TypedDict):
    """Combined benchmark results."""

    validity: ValidityBenchmarkResults
    fcd: FCDBenchmarkResults
    kl_score: float
    "KL score from guacamol (https://arxiv.org/pdf/1811.09621). In [0,1]"
    moses: MosesBenchmarkResults


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

    def benchmark_model(self, model: MoleculeGenerationModel) -> BenchmarkResults:
        """Run the benchmarks on the generated SMILES."""

        generated_smiles = model.generate_molecules(self.num_samples_to_generate)
        if not generated_smiles:
            raise ValueError("No generated SMILES provided for benchmarking.")
        return self.benchmark(generated_smiles)

    def benchmark(self, generated_smiles: list[str | None]) -> BenchmarkResults:
        """Run the benchmarks on the generated SMILES."""
        if len(generated_smiles) < self.num_samples_to_generate:
            raise ValueError(
                f"Expected at least {self.num_samples_to_generate} generated SMILES, but got {len(generated_smiles)}."
            )
        generated_smiles = generated_smiles[: self.num_samples_to_generate]
        generated_smiles = canonicalize_smiles_list(generated_smiles)
        kl_score = self._compute_kl_score(generated_smiles)
        validity_results = self._compute_validity_scores(generated_smiles)
        fcd_results = self._compute_fcd_scores(generated_smiles)

        existing_generated_smiles = [s for s in generated_smiles if s is not None]
        moses_results: MosesBenchmarkResults = {
            "fraction_passing_moses_filters": self.get_fraction_passing_moses_filters(
                generated_smiles
            ),
            "snn_score": self.get_snn_score(generated_smiles),
            "IntDiv": float(internal_diversity(existing_generated_smiles, p=1)),
            "IntDiv2": float(internal_diversity(existing_generated_smiles, p=2)),
            "scaffolds_similarity": self.compute_scaffold_similarity(generated_smiles),
            "fragment_similarity": self.compute_fragment_similarity(generated_smiles),
        }

        return {
            "validity": validity_results,
            "fcd": fcd_results,
            "kl_score": kl_score,
            "moses": moses_results,
        }

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

        # metadata = {"number_samples": self.number_samples, "kl_divs": kldivs}

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = float(sum(partial_scores) / len(partial_scores))
        print("KL divergence score:", score)
        return score

    def get_snn_score(self, generated_smiles: list[str | None]) -> float:
        """Compute the SNN score for the generated SMILES."""
        train_fingerprints = fingerprints(
            self.dataset.get_train_smiles(),
        )
        generated_fingerprints = fingerprints(
            [s for s in generated_smiles if s is not None],
        )
        return float(
            average_agg_tanimoto(
                train_fingerprints, generated_fingerprints, device=self.device
            )
        )

    def get_fraction_passing_moses_filters(
        self, generated_smiles: list[str | None]
    ) -> float:
        """Compute the fraction of generated SMILES that pass the Moses filters."""
        passes = mapper(1)(mol_passes_filters, generated_smiles)
        return float(np.mean(passes))

    def compute_scaffold_similarity(self, generated_smiles: list[str | None]):
        """Compute the scaffold similarity of the generated SMILES."""
        valid_smiles = [
            s for s in generated_smiles if s is not None and is_valid_smiles(s)
        ]
        train_scaffolds = compute_scaffolds(self.dataset.get_train_smiles())
        generated_scaffolds = compute_scaffolds(valid_smiles)
        return float(cos_similarity(train_scaffolds, generated_scaffolds))

    def compute_fragment_similarity(self, generated_smiles: list[str | None]) -> float:
        """Compute the fragment similarity of the generated SMILES."""
        valid_smiles = [
            s for s in generated_smiles if s is not None and is_valid_smiles(s)
        ]
        train_fragments = compute_fragments(self.dataset.get_train_molecules())
        generated_fragments = compute_fragments(valid_smiles)
        return float(cos_similarity(train_fragments, generated_fragments))
