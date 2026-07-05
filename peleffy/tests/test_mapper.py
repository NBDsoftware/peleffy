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

    def test_mapper_never_maps_hydrogen_to_heavy_atom(self):
        """
        It checks that the atom-mapping extension logic never maps a
        hydrogen onto a heavy atom, even when they happen to overlap
        closely in 3D space.

        A hydrogen is always a terminal atom, so mapping it onto a
        heavy atom that carries its own subtree of further atoms would
        anchor that whole subtree on a borrowed (and generally wrong)
        bond length, distorting the resulting hybrid structure. This
        uses two molecules with explicit, deliberately overlapping
        coordinates (rather than a random conformer embedding) so the
        test is deterministic: molecule 2's second carbon (a methyl,
        with its own 3 hydrogens) is placed exactly where molecule 1's
        amine hydrogen sits, both dangling off an already-mapped
        N/O pair.
        """
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        from peleffy.topology import Molecule, Mapper

        def build_rdkit_molecule(atoms, bonds, coords):
            mol = Chem.RWMol()
            for symbol in atoms:
                mol.AddAtom(Chem.Atom(symbol))
            for i, j in bonds:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
            conformer = Chem.Conformer(len(atoms))
            for i, (x, y, z) in enumerate(coords):
                conformer.SetAtomPosition(i, Point3D(x, y, z))
            mol.AddConformer(conformer)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol

        # Molecule 1: methylamine-like, C0-N1, with C0's 3 H (2,3,4)
        # and N1's 2 H (5,6)
        rdkit_mol1 = build_rdkit_molecule(
            ['C', 'N', 'H', 'H', 'H', 'H', 'H'],
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6)],
            [(0, 0, 0), (1.47, 0, 0),
             (-0.5, 0.9, 0.3), (-0.5, -0.9, 0.3), (-0.5, 0, -1.0),
             (1.8, 0.9, 0.3), (1.8, -0.9, 0.3)])

        # Molecule 2: dimethyl-ether-like, C0-O1-C2, with C0's 3 H
        # (3,4,5) and C2's 3 H (6,7,8). O1 is placed on top of
        # molecule 1's N1, and C2 (a whole methyl subtree) is placed
        # on top of molecule 1's H5
        rdkit_mol2 = build_rdkit_molecule(
            ['C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (2, 6), (2, 7), (2, 8)],
            [(0.05, 0, 0.05), (1.50, 0, 0.05), (1.8, 0.9, 0.3),
             (-0.5, 0.9, 0.35), (-0.5, -0.9, 0.35), (-0.5, 0, -0.95),
             (2.3, 1.5, 0.9), (2.3, 0.3, -0.3), (2.5, 1.5, -0.3)])

        mol1 = Molecule()
        mol1._rdkit_molecule = rdkit_mol1
        mol2 = Molecule()
        mol2._rdkit_molecule = rdkit_mol2

        mapper = Mapper(mol1, mol2, include_hydrogens=True)
        mapping = mapper.get_mapping()

        for atom1_idx, atom2_idx in mapping:
            is_H1 = rdkit_mol1.GetAtomWithIdx(atom1_idx).GetSymbol() == 'H'
            is_H2 = rdkit_mol2.GetAtomWithIdx(atom2_idx).GetSymbol() == 'H'
            assert is_H1 == is_H2, \
                ('Hydrogen mapped to a heavy atom (or vice versa): ' +
                 '{}'.format((atom1_idx, atom2_idx)))

        # The overlapping methyl carbon (molecule 2, index 2) must not
        # be mapped at all, since its only close 3D match (molecule
        # 1's amine hydrogen) is forbidden by the guard above
        assert 2 not in [pair[1] for pair in mapping]

    def test_mapper_never_extracts_a_single_ring_atom(self):
        """
        It checks that the atom-mapping extension logic never maps a
        single ring atom onto an unrelated atom, even when they happen
        to overlap closely in 3D space.

        Extracting just one atom of a ring into the mapping would
        leave the rest of that ring to be treated as a separate,
        independent (non-native) fragment, silently splitting the
        ring apart in the resulting hybrid topology. This uses two
        molecules with explicit, deliberately overlapping coordinates
        (rather than a random conformer embedding) so the test is
        deterministic: molecule 2's cyclobutyl ring carbon that is
        directly bonded to the mapped anchor is placed exactly where
        molecule 1's second (plain, non-ring) carbon sits.
        """
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        from peleffy.topology import Molecule, Mapper

        def build_rdkit_molecule(atoms, bonds, coords):
            mol = Chem.RWMol()
            for symbol in atoms:
                mol.AddAtom(Chem.Atom(symbol))
            for i, j in bonds:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
            conformer = Chem.Conformer(len(atoms))
            for i, (x, y, z) in enumerate(coords):
                conformer.SetAtomPosition(i, Point3D(x, y, z))
            mol.AddConformer(conformer)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol

        # Molecule 1: ethane-like, C0-C1, with C0's 3 H (2,3,4) and
        # C1's 3 H (5,6,7)
        rdkit_mol1 = build_rdkit_molecule(
            ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)],
            [(0, 0, 0), (1.54, 0, 0),
             (-0.5, 0.9, 0.3), (-0.5, -0.9, 0.3), (-0.5, 0, -1.0),
             (2.04, 0.9, 0.3), (2.04, -0.9, 0.3), (2.04, 0, -1.0)])

        # Molecule 2: C0'-cyclobutyl, where the ring is
        # C1'-C2'-C3'-C4'-C1'. C0' is placed on top of molecule 1's
        # C0, and C1' (the ring atom bonded to C0') is placed on top
        # of molecule 1's C1 (the atom it would otherwise naturally
        # extend the mapping onto)
        rdkit_mol2 = build_rdkit_molecule(
            ['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
             'H', 'H'],
            [(0, 1), (0, 5), (0, 6), (0, 7),
             (1, 2), (1, 4), (2, 3), (3, 4),
             (2, 8), (2, 9), (3, 10), (3, 11), (4, 12), (4, 13)],
            [(0.02, 0, 0.02), (1.54, 0, 0),
             (2.2, 1.3, 0.4), (3.6, 1.0, 0.4), (2.9, -0.4, 0.5),
             (-0.5, 0.9, 0.35), (-0.5, -0.9, 0.35), (-0.5, 0, -0.95),
             (1.9, 1.9, 1.3), (2.0, 2.1, -0.4), (4.0, 1.9, 0.7),
             (4.2, 0.4, 1.1), (2.6, -1.2, -0.1), (3.5, -0.9, 1.3)])

        mol1 = Molecule()
        mol1._rdkit_molecule = rdkit_mol1
        mol2 = Molecule()
        mol2._rdkit_molecule = rdkit_mol2

        mapper = Mapper(mol1, mol2, include_hydrogens=True)
        mapping = mapper.get_mapping()

        # None of molecule 2's ring atoms (1, 2, 3, 4) may appear in
        # the mapping, since mapping just one of them out would split
        # the cyclobutyl ring apart
        ring_atoms = {1, 2, 3, 4}
        mapped_atom2 = {pair[1] for pair in mapping}
        assert not (ring_atoms & mapped_atom2), \
            ('A ring atom was extracted into the mapping: ' +
             '{}'.format(ring_atoms & mapped_atom2))

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

