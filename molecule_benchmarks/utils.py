import hashlib
import logging
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import psutil
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# from scipy import histogram
from scipy.stats import entropy, gaussian_kde
from tqdm import tqdm

# Mute RDKit logger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def remove_duplicates(smiles_list: Iterable[str | None]) -> List[str]:
    return list(set(s for s in smiles_list if s is not None))


# def is_valid(smiles: str):
#     """
#     Verifies whether a SMILES string corresponds to a valid molecule.

#     Args:
#         smiles: SMILES string

#     Returns:
#         True if the SMILES strings corresponds to a valid, non-empty molecule.
#     """

#     mol = Chem.MolFromSmiles(smiles)

#     return smiles != "" and mol is not None and mol.GetNumAtoms() > 0


def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


def canonicalize_list(
    smiles_list: Iterable[str], include_stereocenters=True
) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """
    canonicalized_smiles = mapper(job_name="Canonicalizing SMILES")(
        partial(canonicalize, include_stereocenters=include_stereocenters), smiles_list
    )
    # canonicalized_smiles = [
    #     canonicalize(smiles, include_stereocenters) for smiles in smiles_list
    # ]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)


def smiles_to_rdkit_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Converts a SMILES string to a RDKit molecule.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        RDKit Mol, None if the SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)

    #  Sanitization check (detects invalid valence)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None

    return mol


def split_charged_mol(smiles: str) -> str:
    if smiles.count(".") > 0:
        largest = ""
        largest_len = -1
        split = smiles.split(".")
        for i in split:
            if len(i) > largest_len:
                largest = i
                largest_len = len(i)
        return largest

    else:
        return smiles


def initialise_neutralisation_reactions():
    patts = (
        # Imidazoles
        ("[n+;H]", "n"),
        # Amines
        ("[N+;!H0]", "N"),
        # Carboxylic acids and alcohols
        ("[$([O-]);!$([O-][#7])]", "O"),
        # Thiols
        ("[S-;X1]", "S"),
        # Sulfonamides
        ("[$([N-;X2]S(=O)=O)]", "N"),
        # Enamines
        ("[$([N-;X2][C,N]=C)]", "N"),
        # Tetrazoles
        ("[n-]", "[nH]"),
        # Sulfoxides
        ("[$([S-]=O)]", "S"),
        # Amides
        ("[$([N-]C=O)]", "N"),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def neutralise_charges(mol, reactions=None):
    replaced = False

    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        Chem.SanitizeMol(mol)
        return mol, True
    else:
        return mol, False


def filter_and_canonicalize(
    smiles: str,
    holdout_set,
    holdout_fps,
    neutralization_rxns,
    tanimoto_cutoff=0.5,
    include_stereocenters=False,
):
    """
    Args:
        smiles: the molecule to process
        holdout_set: smiles of the holdout set
        holdout_fps: ECFP4 fingerprints of the holdout set
        neutralization_rxns: neutralization rdkit reactions
        tanimoto_cutoff: Remove molecules with a higher ECFP4 tanimoto similarity than this cutoff from the set
        include_stereocenters: whether to keep stereocenters during canonicalization

    Returns:
        list with canonical smiles as a list with one element, or a an empty list. This is to perform a flatmap:
    """
    try:
        # Drop out if too long
        if len(smiles) > 200:
            return []
        mol = Chem.MolFromSmiles(smiles)
        # Drop out if invalid
        if mol is None:
            return []
        mol = Chem.RemoveHs(mol)

        # We only accept molecules consisting of H, B, C, N, O, F, Si, P, S, Cl, aliphatic Se, Br, I.
        metal_smarts = Chem.MolFromSmarts(
            "[!#1!#5!#6!#7!#8!#9!#14!#15!#16!#17!#34!#35!#53]"
        )

        has_metal = mol.HasSubstructMatch(metal_smarts)

        # Exclude molecules containing the forbidden elements.
        if has_metal:
            print(f"metal {smiles}")
            return []

        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)

        # Drop out if too long canonicalized:
        if len(canon_smi) > 100:
            return []
        # Balance charges if unbalanced
        if canon_smi.count("+") - canon_smi.count("-") != 0:
            new_mol, changed = neutralise_charges(mol, reactions=neutralization_rxns)
            if changed:
                mol = new_mol
                canon_smi = Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)

        # Get most similar to holdout fingerprints, and exclude too similar molecules.
        max_tanimoto = highest_tanimoto_precalc_fps(mol, holdout_fps)
        if max_tanimoto < tanimoto_cutoff and canon_smi not in holdout_set:
            return [canon_smi]
        else:
            print("Exclude: {} {}".format(canon_smi, max_tanimoto))
    except Exception as e:
        print(e)
    return []


def calculate_internal_pairwise_similarities(
    smiles_list: Collection[str],
) -> np.ndarray:
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    if len(smiles_list) > 10000:
        logger.warning(
            f"Calculating internal similarity on large set of "
            f"SMILES strings ({len(smiles_list)})"
        )

    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities


def calculate_pairwise_similarities(
    smiles_list1: List[str], smiles_list2: List[str]
) -> np.ndarray:
    """
    Computes the pairwise ECFP4 tanimoto similarity of the two smiles containers.

    Returns:
        Pairwise similarity matrix as np.ndarray
    """
    if len(smiles_list1) > 10000 or len(smiles_list2) > 10000:
        logger.warning(
            f"Calculating similarity between large sets of "
            f"SMILES strings ({len(smiles_list1)} x {len(smiles_list2)})"
        )

    mols1 = get_mols(smiles_list1)
    fps1 = get_fingerprints(mols1)

    mols2 = get_mols(smiles_list2)
    fps2 = get_fingerprints(mols2)

    similarities = []

    for fp1 in fps1:
        sims = DataStructs.BulkTanimotoSimilarity(fp1, fps2)

        similarities.append(sims)

    return np.array(similarities)


def get_fingerprints_from_smileslist(smiles_list):
    """
    Converts the provided smiles into ECFP4 bitvectors of length 4096.

    Args:
        smiles_list: list of SMILES strings

    Returns: ECFP4 bitvectors of length 4096.

    """
    return get_fingerprints(get_mols(smiles_list))


def get_fingerprints(mols: Iterable[Chem.Mol], radius=2, length=4096):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]


def get_mols(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    mols = mapper(job_name="Converting SMILES to RDKit Mol")(
        partial(get_mol, sanitize=False), smiles_list
    )
    mols = [mol for mol in mols if mol is not None]
    return mols
    # for i in smiles_list:
    #     try:
    #         mol = Chem.MolFromSmiles(i)
    #         if mol is not None:
    #             yield mol
    #     except Exception as e:
    #         logger.warning(e)


def highest_tanimoto_precalc_fps(mol, fps):
    """

    Args:
        mol: Rdkit molecule
        fps: precalculated ECFP4 bitvectors

    Returns:

    """

    if fps is None or len(fps) == 0:
        return 0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 4096)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp1, fps))

    return sims.max()


def continuous_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray) -> float:
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(
        np.hstack([X_baseline, X_sampled]).min(),
        np.hstack([X_baseline, X_sampled]).max(),
        num=1000,
    )
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10
    return entropy(P, Q)


def discrete_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray) -> float:
    P, bins = np.histogram(X_baseline, bins=10, density=True)
    P += 1e-10
    Q, _ = np.histogram(X_sampled, bins=bins, density=True)
    Q += 1e-10

    return entropy(P, Q)


def calculate_pc_descriptors(
    smiles: Iterable[str], pc_descriptors: List[str]
) -> np.ndarray:
    output = []
    descriptors = mapper(job_name="Computing PC Descriptors")(
        partial(_calculate_pc_descriptors, pc_descriptors=pc_descriptors), smiles
    )
    output = [d for d in descriptors if d is not None]
    return np.array(output)


def _calculate_pc_descriptors(
    smiles: str, pc_descriptors: List[str]
) -> Optional[np.ndarray]:
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(pc_descriptors)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    _fp = calc.CalcDescriptors(mol)
    _fp = np.array(_fp)
    mask = np.isfinite(_fp)
    if (mask == 0).sum() > 0:
        logger.warning(f"{smiles} contains an NAN physchem descriptor")
        _fp[~mask] = 0

    return _fp


def parse_molecular_formula(formula: str) -> List[Tuple[str, int]]:
    """
    Parse a molecular formulat to get the element types and counts.

    Args:
        formula: molecular formula, f.i. "C8H3F3Br"

    Returns:
        A list of tuples containing element types and number of occurrences.
    """
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

    # Convert matches to the required format
    results = []
    for match in matches:
        # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
        count = 1 if not match[1] else int(match[1])
        results.append((match[0], count))

    return results


def is_valid_smiles(smiles: str | None) -> bool:
    """Check if a SMILES string is valid."""
    if smiles is None or not isinstance(smiles, str) or len(smiles) == 0:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
        return False


def filter_valid_smiles(smiles_list: Iterable[str | None]) -> list[str]:
    """Filter a list of SMILES strings to only include valid ones."""
    valid_masks = mapper(job_name="Filtering valid SMILES")(
        is_valid_smiles, smiles_list
    )
    return [s for s, valid in zip(smiles_list, valid_masks) if valid and s is not None]


def download_with_cache(
    url: str, cache_dir: str | Path | None = None, filename: Optional[str] = None
) -> str:
    """
    Download a file from a URL and cache it locally.

    Args:
        url: URL to download the file from
        cache_dir: Directory to store the cached file
        filename: Optional filename to save the file as. If not provided, it will be derived from the URL.

    Returns:
        Path to the downloaded file.
    """
    if cache_dir is None:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        if not content.strip():
            raise ValueError(f"Downloaded file from {url} is empty.")
        return content

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if filename is None:
        hash_val = hashlib.md5(url.encode()).hexdigest()
        # print(f"hash for url {url}:",hash_val)
        filename = f"{hash_val}_{os.path.basename(url)}"

    filepath = os.path.join(cache_dir, filename)

    if not os.path.exists(filepath):
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
    with open(filepath, "r") as f:
        content = f.read()
        if not content.strip():
            raise ValueError(f"Downloaded file {filepath} is empty.")

        return content

def available_cpu_count():
    # 1. Slurm-aware (allocated CPUs)
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get(
        "SLURM_CPUS_PER_TASK"
    )
    if slurm_cpus:
        return int(slurm_cpus)

    # 2. Respect CPU affinity if psutil is available
    try:
        process = psutil.Process()
        if hasattr(process, "cpu_affinity"):
            # psutil.cpu_count() returns the number of logical CPUs
            # cpu_affinity() returns the CPUs that the process is allowed to run on
            # We return the length of the CPU affinity list
            affinity = process.cpu_affinity()
            if affinity:
                return len(affinity)
    except Exception:
        pass

    # 3. Try Python 3.9+'s os.sched_getaffinity (Linux only)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    # 4. Fall back to all visible CPUs (may overcount on clusters)
    return os.cpu_count() or 1  # fallback to 1 if os.cpu_count() returns None

T = TypeVar("T")

def mapper(
    n_jobs: int | None = None,
    job_name: str = "mapping",
    min_length_for_parallel: int = 1000,
) -> Callable[[Callable[..., T], Iterable[Any]], list[T]]:
    """
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = available_cpu_count()
    # print(f"Using {n_jobs} jobs for {job_name}...")
    if n_jobs == 1:

        def _mapper(func: Callable[..., T], iterable: Iterable[Any]) -> list[T]:
            iterable_list = list(iterable)  # Convert to list to get length and reuse
            return list(
                tqdm(
                    map(func, iterable_list),
                    total=len(iterable_list),
                    desc=job_name,
                )
            )

        return _mapper
    if isinstance(n_jobs, int):
        pool = ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"))

        def _mapper(func: Callable[..., T], iterable: Iterable[Any]) -> list[T]:
            # print("args=", (func, iterable))
            len_list = len(list(iterable))
            iterable_list = list(iterable)  # Convert to list to get length and reuse
            if len_list <= min_length_for_parallel:
                # If the list is small, use standard map
                return list(tqdm(map(func, iterable_list), total=len_list, desc=job_name))
            try:
                result = list(
                    tqdm(pool.map(func, iterable_list), total=len_list, desc=f"{job_name} ({n_jobs} jobs)")
                )
            finally:
                pool.shutdown(wait=True)
            return result

        return _mapper
    return n_jobs.map


def get_mol(smiles_or_mol: str | Chem.Mol | None, sanitize=True) -> Chem.Mol | None:
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            if sanitize:
                Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol