from abc import abstractmethod
from typing import Protocol

class MoleculeGenerationModel(Protocol):

    @abstractmethod
    def generate_molecule_batch(self) -> list[str | None]:
        """
        Generate a batch of molecules. This will be called repeatedly until the desired number of molecules is generated.
        The batch size can be decided by the implementation and may vary between calls.

        Returns:
            A list of SMILES strings representing the generated molecules. If a sample cannot be converted to SMILES,
            it should return None for that sample.
        """
        pass


    def generate_molecules(self, num_molecules: int) -> list[str | None]:
        """
        Generate a specified number of molecules.

        Args:
            num_molecules: The number of molecules to generate.

        Returns:
            A list of SMILES strings representing the generated molecules. If a sample cannot be converted to SMILES,
            it returns None for that sample. The list will contain exactly `num_molecules`.
        """
        smiles_list: list[str | None] = []
        while len(smiles_list) < num_molecules:
            batch = self.generate_molecule_batch()
            smiles_list.extend(batch)
            if len(smiles_list) > num_molecules:
                smiles_list = smiles_list[:num_molecules]
        return smiles_list

class DummyMoleculeGenerationModel(MoleculeGenerationModel):
    """
    A dummy model that generates a fixed set of SMILES strings for testing purposes.
    """

    def generate_molecule_batch(self) -> list[str | None]:
        return [
            "C1=CC=CC=C1",  # Benzene
            "C1=CC=CN=C1",  # Pyridine
            "C1=CC=CO=C1",  # Furan
            None,           # Invalid SMILES
        ]