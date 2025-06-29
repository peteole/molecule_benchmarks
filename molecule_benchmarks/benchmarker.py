import math
import random
from typing import TypedDict

import numpy as np
from fcd import (  # type: ignore
    calculate_frechet_distance,
    get_predictions,
    load_ref_model,
)
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
    filter_valid_smiles,
    is_valid_smiles,
)


class ValidityBenchmarkResults(TypedDict):
    num_molecules_generated: int
    valid_fraction: float
    valid_and_unique_fraction: float
    unique_fraction: float
    unique_fraction_at_1000: float
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
    "KL score from guacamol (https://arxiv.org/pdf/1811.09621). In [0,1], higher is better."
    moses: MosesBenchmarkResults






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
        print("Precomputing validation set statistics...")
        self.val_mu, self.val_sigma = self._compute_fcd_mu_sigma(
            self.dataset.get_validation_smiles(), max_samples=num_samples_to_generate
        )
        self.val_fingerprints_morgan = fingerprints(
            self.dataset.get_validation_smiles(),
            fp_type="morgan",
        )
        self.val_scaffolds = compute_scaffolds(self.dataset.get_validation_smiles())
        self.val_fragments = compute_fragments(self.dataset.get_validation_smiles())

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
        valid_smiles = filter_valid_smiles(generated_smiles)
        kl_score = self._compute_kl_score(generated_smiles)
        validity_results = self._compute_validity_scores(generated_smiles, valid_smiles)
        fcd_results = self._compute_fcd_scores(generated_smiles, generated_valid_smiles=valid_smiles)
        moses_results: MosesBenchmarkResults = {
            "fraction_passing_moses_filters": self.get_fraction_passing_moses_filters(
                valid_smiles
            ),
            "snn_score": self.get_snn_score(valid_smiles),
            "IntDiv": float(internal_diversity(valid_smiles, p=1)),
            "IntDiv2": float(internal_diversity(valid_smiles, p=2)),
            "scaffolds_similarity": self.compute_scaffold_similarity(valid_smiles),
            "fragment_similarity": self.compute_fragment_similarity(valid_smiles),
        }

        return {
            "validity": validity_results,
            "fcd": fcd_results,
            "kl_score": kl_score,
            "moses": moses_results,
        }

    def _compute_validity_scores(
        self, generated_smiles: list[str | None], generated_valid_smiles: list[str]
    ) -> ValidityBenchmarkResults:
        unique: set[str|None] = set(generated_smiles)
        existing = set(self.dataset.train_smiles)
        valid_and_unique_fraction = (
            len(set(generated_valid_smiles)) / len(generated_smiles) if generated_smiles else 0.0
        )
        valid_fraction = len(generated_valid_smiles) / len(generated_smiles) if generated_smiles else 0.0
        unique_fraction = (
            len(unique) / len(generated_smiles) if generated_smiles else 0.0
        )
        unique_at_1000 = set(generated_smiles[:1000])
        unique_fraction_at_1000 = len(unique_at_1000) / 1000 if len(generated_smiles) >= 1000 else -1
        unique_and_novel_fraction = (
            len(unique - existing) / len(generated_smiles) if generated_smiles else 0.0
        )
        valid_and_unique_and_novel_fraction = (
            len(set(generated_valid_smiles) - existing) / len(generated_smiles)
            if generated_smiles
            else 0.0
        )
        return {
            "num_molecules_generated": len(generated_smiles),
            "valid_fraction": valid_fraction,
            "valid_and_unique_fraction": valid_and_unique_fraction,
            "unique_fraction": unique_fraction,
            "unique_fraction_at_1000": unique_fraction_at_1000,
            "unique_and_novel_fraction": unique_and_novel_fraction,
            "valid_and_unique_and_novel_fraction": valid_and_unique_and_novel_fraction,
        }
    def _compute_fcd_mu_sigma(
        self, present_smiles: list[str], max_samples: int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(present_smiles) == 0:
            raise ValueError("No valid SMILES provided for FCD computation.")
        random.seed(42)  # For reproducibility
        smiles_to_use = random.sample(present_smiles, min(len(present_smiles), max_samples))
        print(f"Computing FCD mu and sigma with {len(smiles_to_use)} / {len(present_smiles)} samples")
        model = load_ref_model()
        chemnet_activations = get_predictions(
            model, smiles_to_use, device=self.device
        )
        mu=np.mean(chemnet_activations, axis=0)
        sigma=np.cov(chemnet_activations.T)
        return mu, sigma

    def _compute_fcd_scores(
        self, generated_smiles: list[str | None], generated_valid_smiles: list[str]
    ) -> FCDBenchmarkResults:
        """Compute the Fréchet ChemNet Distance (FCD) scores for the generated SMILES. Removes any None-type smiles."""
        print("Computing FCD scores for the generated SMILES...")
        present_smiles = [
            smiles for smiles in generated_smiles if smiles is not None and smiles != ""
        ]
        if len(present_smiles) == 0:
            return {
                "fcd": -1,
                "fcd_valid": -1,
                "fcd_normalized": 1.0,
                "fcd_valid_normalized": 1.0,
            }
        mu, sigma = self._compute_fcd_mu_sigma(
            present_smiles, max_samples=self.num_samples_to_generate
        )
        
        fcd_score = calculate_frechet_distance(
            mu1=mu,
            sigma1=sigma,
            mu2=self.val_mu,
            sigma2=self.val_sigma,
        )
        if len(generated_valid_smiles) == 0:
            return {
                "fcd": fcd_score,
                "fcd_valid": -1,
                "fcd_normalized": math.exp(-0.2 * fcd_score),
                "fcd_valid_normalized": 1.0,
            }
        mu_valid, sigma_valid = self._compute_fcd_mu_sigma(
            generated_valid_smiles, max_samples=self.num_samples_to_generate
        )
        fcd_valid_score = calculate_frechet_distance(
            mu1=mu_valid,
            sigma1=sigma_valid,
            mu2=self.val_mu,
            sigma2=self.val_sigma,
        )
        return {
            "fcd": fcd_score,
            "fcd_valid": fcd_valid_score,
            "fcd_normalized": math.exp(-0.2 * fcd_score),
            "fcd_valid_normalized": math.exp(-0.2 * fcd_valid_score),
        }

    def _compute_kl_score(self, generated_smiles: list[str | None]) -> float:
        """Compute the KL divergence score for the generated SMILES. Code is from guacamol:
        https://github.com/BenevolentAI/guacamol/blob/master/guacamol/distribution_learning_benchmark.py#L161"""
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
        if len(d_sampled) == 0:
            return 0.0
        
        random.seed(42)  # For reproducibility
        # pairwise similarity
        random_train_samples = random.sample(
            self.dataset.get_train_smiles(), min(len(generated_smiles_valid), len(self.dataset.get_train_smiles()))
        )
        d_chembl = calculate_pc_descriptors(
            random_train_samples, pc_descriptor_subset
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
        chembl_sim = calculate_internal_pairwise_similarities(
            random_train_samples
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

    def get_snn_score(self, generated_valid_smiles: list[str ]) -> float:
        """Compute the SNN score for the generated SMILES."""
        if len(generated_valid_smiles) == 0:
            return 0.0
        
        generated_fingerprints = fingerprints(
            generated_valid_smiles, fp_type="morgan"
        )
        return float(
            average_agg_tanimoto(
                self.val_fingerprints_morgan, generated_fingerprints, device=self.device
            )
        )

    def get_fraction_passing_moses_filters(
        self, generated_valid_smiles: list[str]
    ) -> float:
        """Compute the fraction of generated SMILES that pass the Moses filters."""
        passes = mapper(job_name="Moses filters")(mol_passes_filters, generated_valid_smiles)
        return float(np.mean(passes))

    def compute_scaffold_similarity(self, generated_valid_smiles: list[str]):
        """Compute the scaffold similarity of the generated SMILES."""
        # valid_smiles = [
        #     s for s in generated_smiles if s is not None and is_valid_smiles(s)
        # ]
        # if len(valid_smiles) == 0:
        #     return 0.0
        generated_scaffolds = compute_scaffolds(generated_valid_smiles)
        return float(cos_similarity(self.val_scaffolds, generated_scaffolds))

    def compute_fragment_similarity(self, generated_valid_smiles: list[str]) -> float:
        """Compute the fragment similarity of the generated SMILES."""
        #valid_smiles = filter_valid_smiles(generated_smiles)
        # if len(generated_smiles) == 0:
        #     return 0.0
        
        generated_fragments = compute_fragments(generated_valid_smiles)
        return float(cos_similarity(self.val_fragments, generated_fragments))
