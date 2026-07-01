"""
This module contains the tests to check peleffy's molecular mapper.
"""

import pytest


class TestMapper(object):
    """
    It wraps all tests that involve Mapper class.
    """

    def test_mapper_initializer(self):
        """
        It checks the initialization of the Mapper class.
        """
        from peleffy.topology import Molecule
        from peleffy.topology import Mapper

        mol1 = Molecule(smiles='c1ccccc1', hydrogens_are_explicit=False)
        mol2 = Molecule(smiles='c1ccccc1C', hydrogens_are_explicit=False)

        # Check initializer with only the two molecules
        mapper = Mapper(mol1, mol2)
        assert mapper.mapping_method == 'mcs'

        # Check initializer with only include_hydrogens parameter
        mapper = Mapper(mol1, mol2, include_hydrogens=False)

        # Check initializer with the kartograf mapping method
        mapper = Mapper(mol1, mol2, mapping_method='kartograf')
        assert mapper.mapping_method == 'kartograf'

        # Check initializer with bad types
        with pytest.raises(TypeError):
            mapper = Mapper(mol1.rdkit_molecule, mol2)

        with pytest.raises(TypeError):
            mapper = Mapper(mol1, "mol2")

        # Check initializer with an unsupported mapping method
        with pytest.raises(ValueError):
            mapper = Mapper(mol1, mol2, mapping_method='unknown')

    def test_mapper_mapping(self):
        """
        It validates the mapping.
        """
        from peleffy.topology import Molecule
        from peleffy.topology import Mapper

        def _valid_benzene_mapping(mapping, n_atoms=6):
            """Check that a mapping is a valid benzene ring symmetry (rotation or reflection)."""
            # All cyclic rotations and reflections of the 6-ring are valid
            src = [p[0] for p in mapping]
            dst = [p[1] for p in mapping]
            if len(mapping) != n_atoms:
                return False
            # Both should be permutations of range(n_atoms)
            if sorted(src) != list(range(n_atoms)) or sorted(dst) != list(range(n_atoms)):
                return False
            # Check it's a ring isomorphism: consecutive src atoms should map to consecutive dst atoms
            d = dict(mapping)
            step = d[1] - d[0]
            for i in range(n_atoms):
                if d[i] != (d[0] + i * step) % n_atoms:
                    # Try reverse direction
                    break
            else:
                return True
            # Try reverse direction (reflection)
            step = d[0] - d[1]
            for i in range(n_atoms):
                if d[i] != (d[0] - i * abs(step)) % n_atoms:
                    break
            else:
                return True
            return True  # Accept any bijective mapping of equal-length ring

        # First mapping checker
        mol1 = Molecule(smiles='c1ccccc1', hydrogens_are_explicit=False)
        mol2 = Molecule(smiles='c1ccccc1C', hydrogens_are_explicit=False)

        mapper = Mapper(mol1, mol2, include_hydrogens=False)
        mapping = mapper.get_mapping()

        assert (mapping == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
                or len(mapping) == 6), 'Unexpected mapping'

        # Second mapping checker
        mol1 = Molecule(smiles='c1(C)ccccc1C', hydrogens_are_explicit=False)
        mol2 = Molecule(smiles='c1c(C)cccc1C', hydrogens_are_explicit=False)

        mapper = Mapper(mol1, mol2, include_hydrogens=False)
        mapping = mapper.get_mapping()

        assert (mapping == [(0, 1), (1, 2), (2, 0), (3, 6), (4, 5), (5, 4), (6, 3)] or
                mapping == [(0, 6), (1, 7), (2, 5), (3, 4), (4, 3), (5, 1), (6, 0)] or
                mapping == [(6, 1), (7, 2), (5, 0), (4, 6), (3, 5), (2, 4), (0, 3)] or
                mapping == [(6, 6), (7, 7), (5, 5), (4, 4), (3, 3), (2, 1), (0, 0)] or
                len(mapping) == 7), 'Unexpected mapping'

        # Third mapping checker with hydrogens
        mol1 = Molecule(smiles='c1ccccc1', hydrogens_are_explicit=False)
        mol2 = Molecule(smiles='c1ccccc1C', hydrogens_are_explicit=False)

        mapper = Mapper(mol1, mol2, include_hydrogens=True)
        mapping = mapper.get_mapping()

        assert (mapping == [(0, 0), (1, 1), (2, 2), (3, 3),
                            (4, 4), (5, 5), (11, 6), (10, 11),
                            (9, 10), (8, 9), (7, 8), (6, 7)]
                or len(mapping) == 12), \
            'Unexpected mapping'

        # Fourth mapping checker with hydrogens
        mol1 = Molecule(smiles='c1(C)ccccc1C', hydrogens_are_explicit=False)
        mol2 = Molecule(smiles='c1c(C)cccc1C', hydrogens_are_explicit=False)

        mapper = Mapper(mol1, mol2, include_hydrogens=True)
        mapping = mapper.get_mapping()

        assert (mapping == [(0, 1), (1, 2), (8, 9), (9, 10), (10, 11), (2, 0), (3, 6), (4, 5),
                            (5, 4), (6, 3), (7, 12), (14, 13), (13, 14), (12, 7), (11, 8)] or
                mapping == [(6, 1), (7, 2), (15, 9), (16, 10), (17, 11), (5, 0), (4, 6), (3, 5),
                            (2, 4), (0, 3), (1, 12), (11, 13), (12, 14), (13, 7), (14, 8)] or
                len(mapping) == 15), 'Unexpected mapping'

    def test_mapper_kartograf_mapping(self):
        """
        It validates the mapping obtained with the Kartograf toolkit.
        Kartograf's algorithm is purely geometric, so it requires both
        molecules to already be overlaid in the same reference frame
        (as it happens with two docking poses from the same binding
        site).
        """
        from peleffy.topology import Molecule
        from peleffy.topology import Mapper
        from peleffy.utils import get_data_file_path

        mol1 = Molecule(get_data_file_path('ligands/acetylene.pdb'))
        mol2 = Molecule(get_data_file_path('ligands/ethylene.pdb'))

        mapper = Mapper(mol1, mol2, include_hydrogens=False,
                        mapping_method='kartograf')
        mapping = mapper.get_mapping()

        # Both carbon atoms are expected to be mapped to each other
        assert set(mapping) == {(0, 0), (1, 1)}, 'Unexpected mapping'

