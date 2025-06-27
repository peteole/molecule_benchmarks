API Reference
=============

This section provides detailed documentation for all classes, functions, and modules in the Molecule Benchmarks package.

Main Classes
------------

.. currentmodule:: molecule_benchmarks

Benchmarker
~~~~~~~~~~~

.. autoclass:: Benchmarker
   :members:
   :undoc-members:
   :show-inheritance:

SmilesDataset
~~~~~~~~~~~~~

.. autoclass:: SmilesDataset
   :members:
   :undoc-members:
   :show-inheritance:

Model Interface
---------------

.. currentmodule:: molecule_benchmarks.model

MoleculeGenerationModel
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MoleculeGenerationModel
   :members:
   :undoc-members:
   :show-inheritance:

DummyMoleculeGenerationModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DummyMoleculeGenerationModel
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Module
--------------

.. currentmodule:: molecule_benchmarks.dataset

.. automodule:: molecule_benchmarks.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: load_smiles
.. autofunction:: canonicalize_smiles_list

Benchmarker Module
------------------

.. currentmodule:: molecule_benchmarks.benchmarker

.. automodule:: molecule_benchmarks.benchmarker
   :members:
   :undoc-members:
   :show-inheritance:

Type Definitions
~~~~~~~~~~~~~~~~

.. autoclass:: ValidityBenchmarkResults
   :members:

.. autoclass:: FCDBenchmarkResults
   :members:

.. autoclass:: MosesBenchmarkResults
   :members:

.. autoclass:: BenchmarkResults
   :members:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: is_valid_smiles

Moses Metrics Module
--------------------

.. currentmodule:: molecule_benchmarks.moses_metrics

.. automodule:: molecule_benchmarks.moses_metrics
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: mol_passes_filters
.. autofunction:: internal_diversity
.. autofunction:: compute_scaffolds
.. autofunction:: compute_fragments
.. autofunction:: cos_similarity
.. autofunction:: average_agg_tanimoto

RDKit Metrics Module
--------------------

.. currentmodule:: molecule_benchmarks.rdkit_metrics

.. automodule:: molecule_benchmarks.rdkit_metrics
   :members:
   :undoc-members:
   :show-inheritance:

BasicMolecularMetrics
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BasicMolecularMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Molecular Building Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: build_molecule
.. autofunction:: build_molecule_with_partial_charges
.. autofunction:: mol2smiles
.. autofunction:: check_valency
.. autofunction:: check_stability

Utilities Module
----------------

.. currentmodule:: molecule_benchmarks.utils

.. automodule:: molecule_benchmarks.utils
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_pc_descriptors
.. autofunction:: calculate_internal_pairwise_similarities
.. autofunction:: continuous_kldiv
.. autofunction:: discrete_kldiv

Chemical Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: neutralise_charges
.. autofunction:: initialise_neutralisation_reactions

Constants and Variables
-----------------------

.. currentmodule:: molecule_benchmarks.rdkit_metrics

Chemical Constants
~~~~~~~~~~~~~~~~~~

.. autodata:: allowed_bonds
   :annotation: Dict[str, Union[int, List[int]]]

   Dictionary mapping element symbols to their allowed bond counts.

.. autodata:: bond_dict
   :annotation: List[Optional[rdkit.Chem.rdchem.BondType]]

   List mapping bond type indices to RDKit bond types.

.. autodata:: ATOM_VALENCY
   :annotation: Dict[int, int]

   Dictionary mapping atomic numbers to standard valencies.

Exceptions
----------

The package uses standard Python exceptions. Common exceptions you might encounter:

* ``ValueError``: Raised when invalid parameters are provided
* ``FileNotFoundError``: Raised when dataset files cannot be found
* ``ImportError``: Raised when optional dependencies are missing

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from molecule_benchmarks import Benchmarker, SmilesDataset

   # Load dataset
   dataset = SmilesDataset.load_qm9_dataset(subset_size=1000)

   # Create benchmarker
   benchmarker = Benchmarker(dataset, num_samples_to_generate=100)

   # Benchmark SMILES
   results = benchmarker.benchmark(generated_smiles)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from molecule_benchmarks.model import MoleculeGenerationModel

   class MyModel(MoleculeGenerationModel):
       def generate_molecule_batch(self):
           return ["CCO", "CC(=O)O", None]

   model = MyModel()
   results = benchmarker.benchmark_model(model)

Dataset Creation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # From files
   dataset = SmilesDataset(
       train_smiles="train.txt",
       validation_smiles="valid.txt"
   )

   # From lists
   dataset = SmilesDataset(
       train_smiles=["CCO", "CC(=O)O"],
       validation_smiles=["c1ccccc1"]
   )

   # Built-in datasets
   dataset = SmilesDataset.load_moses_dataset(fraction=0.1)

Notes
-----

* All SMILES strings are automatically canonicalized using RDKit
* Invalid molecules should be represented as ``None`` in generated lists
* GPU acceleration is available for FCD calculations when PyTorch with CUDA is installed
* The package uses multiprocessing for efficient computation where applicable
