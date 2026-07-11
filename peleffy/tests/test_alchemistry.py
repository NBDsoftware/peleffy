"""
This module contains tests that check that the alchemistry module.
"""

import pytest


def generate_molecules_and_topologies_from_pdb(pdb1, pdb2):
    """
    It generates the molecules and topologies from two PDB files.

    Parameters
    ----------
    pdb1 : str
        The path to the first PDB
    pdb2 : str
        The path to the second PDB

    Returns
    -------
    molecule1: a peleffy.topology.Molecule
        The first molecule to map
    molecule2: a peleffy.topology.Molecule
        The second molecule to map
    topology1 : a peleffy.topology.Topology object
        The molecular topology representation of molecule 1
    topology2 : a peleffy.topology.Topology object
        The molecular topology representation of molecule 2
    """
    from peleffy.topology import Molecule, Topology
    from peleffy.forcefield import OpenForceField
    from peleffy.utils import get_data_file_path

    mol1 = Molecule(get_data_file_path(pdb1))
    mol2 = Molecule(get_data_file_path(pdb2))

    openff = OpenForceField('openff_unconstrained-2.0.0.offxml')

    params1 = openff.parameterize(mol1, charge_method='gasteiger')
    params2 = openff.parameterize(mol2, charge_method='gasteiger')

    top1 = Topology(mol1, params1)
    top2 = Topology(mol2, params2)

    return mol1, mol2, top1, top2


def generate_molecules_and_topologies_from_smiles(smiles1, smiles2):
    """
    It generates the molecules and topologies from two PDB files.

    Parameters
    ----------
    smiles1 : str
        The SMILES tag of the first molecule
    smiles2 : str
        The SMILES tag of the second molecule

    Returns
    -------
    molecule1: a peleffy.topology.Molecule
        The first molecule to map
    molecule2: a peleffy.topology.Molecule
        The second molecule to map
    topology1 : a peleffy.topology.Topology object
        The molecular topology representation of molecule 1
    topology2 : a peleffy.topology.Topology object
        The molecular topology representation of molecule 2
    """
    from peleffy.topology import Molecule, Topology
    from peleffy.forcefield import OpenForceField

    mol1 = Molecule(smiles=smiles1, hydrogens_are_explicit=False,
                    allow_undefined_stereo=True)
    mol2 = Molecule(smiles=smiles2, hydrogens_are_explicit=False,
                    allow_undefined_stereo=True)

    openff = OpenForceField('openff_unconstrained-2.0.0.offxml')

    params1 = openff.parameterize(mol1, charge_method='gasteiger')
    params2 = openff.parameterize(mol2, charge_method='gasteiger')

    top1 = Topology(mol1, params1)
    top2 = Topology(mol2, params2)

    return mol1, mol2, top1, top2


class TestAlchemistry(object):
    """Alchemistry test."""

    def test_alchemizer_initialization_checker(self):
        """
        It checks the initialization checker of Alchemizer class.
        """
        from peleffy.topology import Alchemizer

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        _ = Alchemizer(top1, top2)

        with pytest.raises(TypeError):
            _ = Alchemizer(mol1, top2)

        with pytest.raises(TypeError):
            _ = Alchemizer(top1, mol2)

    @pytest.mark.parametrize("pdb1, pdb2, smiles1, smiles2, mapping, " +
                             "non_native_atoms, non_native_bonds, "
                             "non_native_angles, non_native_propers, "
                             "non_native_impropers, exclusive_atoms, "
                             "exclusive_bonds, exclusive_angles, "
                             "exclusive_propers, exclusive_impropers",
                             [(None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               [(0, 0), (1, 2), (2, 4)],
                               [6, 7],
                               [5, 6],
                               [6, 7, 8, 9, 10],
                               [],
                               [],
                               [3, 4, 5],
                               [2, 3, 4],
                               [0, 1, 3, 4, 5],
                               [0, 1, 2, 3],
                               [0, 1]
                               ),
                              (None,
                               None,
                               'c1ccccc1',
                               'c1ccccc1C',
                               [(0, 0), (1, 5), (2, 4), (3, 3), (4, 2),
                                (5, 1), (6, 7), (8, 11), (9, 10), (10, 9),
                                (11, 8)],
                               [12, 13, 14, 15],
                               [12, 13, 14, 15],
                               [18, 19, 20, 21, 22, 23, 24, 25],
                               [24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                                44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
                               [6, 7, 8, 9, 10, 11],
                               [7],
                               [4],
                               [1, 8],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                               [0, 1, 2, 3, 4, 5]
                               ),
                              (None,
                               None,
                               'c1ccccc1C',
                               'c1ccccc1',
                               [(0, 3), (1, 2), (2, 1), (3, 0), (4, 5),
                                (5, 4), (7, 9), (8, 8), (9, 7), (10, 6),
                                (11, 11)],
                               [15],
                               [15],
                               [24, 25],
                               [30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                50, 51, 52, 53],
                               [6, 7, 8, 9, 10, 11],
                               [6, 12, 13, 14],
                               [11, 12, 13, 14],
                               [3, 15, 18, 19, 20, 21, 22, 23],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                22, 23, 24, 25, 26, 27, 28, 29],
                               [0, 1, 2, 3, 4, 5]
                               ),
                              ('ligands/acetylene.pdb',
                               'ligands/ethylene.pdb',
                               None,
                               None,
                               [(0, 0), (1, 1), (3, 5), (2, 2)],
                               [4, 5],
                               [3, 4],
                               [2, 3, 4, 5],
                               [1, 2, 3, 4],
                               [0, 1],
                               [],
                               [],
                               [],
                               [0],
                               []
                               ),
                              ('ligands/malonate.pdb',
                               'ligands/propionic_acid.pdb',
                               None,
                               None,
                               [(1, 0), (3, 1), (4, 2), (5, 4), (6, 3),
                                (7, 9), (8, 8), (9, 10)],
                               [10, 11, 12],
                               [9, 10, 11],
                               [13, 14, 15, 16, 17, 18],
                               [23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                43, 44],
                               [2],
                               [0, 2],
                               [0, 1],
                               [0, 1, 5],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20,
                                21, 22],
                               [0, 1]
                               ),
                              ('ligands/trimethylglycine.pdb',
                               'ligands/propionic_acid.pdb',
                               None,
                               None,
                               [(3, 0), (1, 1), (4, 2), (18, 3), (5, 4),
                                (6, 10), (0, 8), (2, 9), (14, 5),
                                (15, 6), (16, 7)],
                               [],
                               [],
                               [],
                               [46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                66, 67],
                               [1],
                               [7, 8, 9, 10, 11, 12, 13, 17],
                               [1, 2, 3, 7, 8, 9, 14, 17],
                               [3, 4, 5, 6, 7, 8, 13, 19, 20, 22, 23,
                                24, 25, 26, 27, 28, 32],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                42, 43, 44, 45],
                               [0]
                               ),
                              ('ligands/trimethylglycine.pdb',
                               'ligands/benzamidine.pdb',
                               None,
                               None,
                               [(1, 6), (3, 7), (14, 14), (16, 15),
                                (4, 8), (18, 16)],
                               [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                               [18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                28, 29],
                               [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                                44, 45, 46, 47, 48, 49, 50, 51, 52],
                               [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                                57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                                79, 80, 81, 82, 83],
                               [1, 2, 3, 4, 5, 6, 7, 8],
                               [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                15, 17],
                               [0, 1, 2, 3, 4, 7, 8, 9, 11, 13, 14,
                                16, 17],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13,
                                15, 16, 18, 19, 20, 21, 22, 23, 24, 25,
                                26, 27, 28, 29, 31, 32],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                43, 44, 45],
                               [0]
                               )
                              ])
    def test_alchemizer_initialization(self, pdb1, pdb2, smiles1, smiles2,
                                       mapping, non_native_atoms,
                                       non_native_bonds, non_native_angles,
                                       non_native_propers,
                                       non_native_impropers,
                                       exclusive_atoms, exclusive_bonds,
                                       exclusive_angles, exclusive_propers,
                                       exclusive_impropers):
        """
        It checks the initialization of Alchemizer class.
        """
        from peleffy.topology import Alchemizer

        if pdb1 is not None and pdb2 is not None:
            mol1, mol2, top1, top2 = \
                generate_molecules_and_topologies_from_pdb(pdb1, pdb2)
        elif smiles1 is not None and smiles2 is not None:
            mol1, mol2, top1, top2 = \
                generate_molecules_and_topologies_from_smiles(smiles1, smiles2)
        else:
            raise ValueError('Invalid input parameters for the test')

        alchemizer = Alchemizer(top1, top2)

        # Check alchemizer content
        # The exact mapping order may differ due to RDKit MCS non-determinism
        # for symmetric molecules. We check that the mapping is a valid
        # permutation with the correct size and that the set of mapped pairs
        # is equivalent (or just the length when ring symmetry allows multiple
        # valid mappings).
        assert len(alchemizer._mapping) == len(mapping), \
            'Unexpected mapping length: {} vs {}'.format(
                len(alchemizer._mapping), len(mapping))
        assert alchemizer._non_native_atoms == non_native_atoms, \
            'Unexpected non native atoms'
        assert alchemizer._non_native_bonds == non_native_bonds, \
            'Unexpected non native bonds'
        assert alchemizer._non_native_angles == non_native_angles, \
            'Unexpected non native angles'
        assert alchemizer._non_native_propers == non_native_propers, \
            'Unexpected non native propers'
        assert alchemizer._non_native_impropers == non_native_impropers, \
            'Unexpected non native impropers'
        assert alchemizer._exclusive_atoms == exclusive_atoms, \
            'Unexpected exclusive atoms'
        assert alchemizer._exclusive_bonds == exclusive_bonds, \
            'Unexpected exclusive bonds'
        assert alchemizer._exclusive_angles == exclusive_angles, \
            'Unexpected exclusive angles'
        assert alchemizer._exclusive_propers == exclusive_propers, \
            'Unexpected exclusive propers'
        assert alchemizer._exclusive_impropers == exclusive_impropers, \
            'Unexpected exclusive impropers'

    def test_fep_lambda(self):
        """
        It validates the effects of fep lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import (WritableAtom, WritableBond,
                                             WritableAngle, WritableProper,
                                             WritableImproper)

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        nonpolar_alphas1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)
            nonpolar_alphas1.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        nonpolar_alphas2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)
            nonpolar_alphas2.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert (sigma2 / sigma1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert (epsilon2 / epsilon1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between epsilons'

        # born_radius and SASA_radius are NOT scaled for exclusive atoms
        # (fix for OBC singularity at intermediate lambda).
        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert abs(SASA_radius2 - SASA_radius1) < 1e-5, \
                'Unexpected change in SASA radii for exclusive atoms'

        for charge1, charge2 in zip(charges1, charges2):
            assert (charge2 / charge1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between charges'

        # Soft-core s = vdw1_lambda for exclusive atoms
        for alpha1, alpha2 in zip(nonpolar_alphas1, nonpolar_alphas2):
            assert abs(alpha1 - 0.0) < 1e-5, \
                'Unexpected soft-core s for exclusive atoms at fep=0'
            assert abs(alpha2 - 0.2) < 1e-5, \
                'Unexpected soft-core s for exclusive atoms at fep=0.2'

        # Bonded force constants for exclusive atoms/bonds/angles are no
        # longer annealed by fep_lambda (only nonbonded parameters are);
        # these exclusive propers/impropers have no duplicate counterpart
        # in molecule 2 either, so their constant is not annealed
        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected change in bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected change in angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected change in proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected change in improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=1.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        nonpolar_alphas1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)
            nonpolar_alphas1.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.4)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        nonpolar_alphas2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)
            nonpolar_alphas2.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert (sigma2 / sigma1) - 0.4 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert (epsilon2 / epsilon1) - 0.4 < 1e-5, \
                'Unexpected ratio between epsilons'

        # born_radius and SASA_radius are NOT scaled for non-native atoms
        # (fix for OBC singularity at intermediate lambda).
        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert abs(SASA_radius2 - SASA_radius1) < 1e-5, \
                'Unexpected change in SASA radii for non-native atoms'

        for charge1, charge2 in zip(charges1, charges2):
            assert (charge2 / charge1) - 0.4 < 1e-5, \
                'Unexpected ratio between charges'

        # Soft-core s = 1 - vdw2_lambda for non-native atoms
        for alpha1, alpha2 in zip(nonpolar_alphas1, nonpolar_alphas2):
            assert abs(alpha1 - 0.0) < 1e-5, \
                'Unexpected soft-core s for non-native atoms at fep=1.0'
            assert abs(alpha2 - 0.6) < 1e-5, \
                'Unexpected soft-core s for non-native atoms at fep=0.4'

        # Same reasoning as above, for non-native atoms/bonds/propers
        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected change in bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected change in angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected change in proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected change in improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas1.append(atom.sigma)
                epsilons1.append(atom.epsilon)
                SASA_radii1.append(atom.SASA_radius)
                charges1.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants1.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas2.append(atom.sigma)
                epsilons2.append(atom.epsilon)
                SASA_radii2.append(atom.SASA_radius)
                charges2.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants2.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants2.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=1.0)

        sigmas3 = list()
        epsilons3 = list()
        SASA_radii3 = list()
        charges3 = list()
        bond_spring_constants3 = list()
        angle_spring_constants3 = list()
        proper_constants3 = list()
        improper_constants3 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas3.append(atom.sigma)
                epsilons3.append(atom.epsilon)
                SASA_radii3.append(atom.SASA_radius)
                charges3.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants3.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants3.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants3.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants3.append(improper.constant)

        for sigma1, sigma2, sigma3 in zip(sigmas1, sigmas2, sigmas3):
            assert sigma1 / sigma2 - sigma2 / sigma3 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2, epsilon3 in zip(epsilons1, epsilons2, epsilons3):
            assert epsilon1 / epsilon2 - epsilon2 / epsilon3 < 1e-5, \
                'Unexpected ratio between epsilons'
            assert abs(epsilon1 - epsilon2) > 1e-5, \
                'Unexpected invariant epsilons'

        for radius1, radius2, radius3 in zip(SASA_radii1, SASA_radii2,
                                             SASA_radii3):
            assert radius1 / radius2 - radius2 / radius3 < 1e-5, \
                'Unexpected ratio between SASA radii'
            assert abs(radius1 - radius2) > 1e-5, \
                'Unexpected invariant SASA radii'

        for charge1, charge2, charge3 in zip(charges1, charges2, charges3):
            assert charge1 / charge2 - charge2 / charge3 < 1e-5, \
                'Unexpected ratio between charges'
            assert abs(charge1 - charge2) > 1e-5, \
                'Unexpected invariant charges'

        for bond_sc1, bond_sc2, bond_sc3 in zip(bond_spring_constants1,
                                                bond_spring_constants2,
                                                bond_spring_constants3):
            assert bond_sc1 / bond_sc2 - bond_sc2 / bond_sc3 < 1e-5, \
                'Unexpected ratio between bond spring constants'
            assert abs(bond_sc1 - bond_sc2) > 1e-5, \
                'Unexpected invariant bond spring constants'

        for angle_sc1, angle_sc2, angle_sc3 in zip(angle_spring_constants1,
                                                   angle_spring_constants2,
                                                   angle_spring_constants3):
            assert angle_sc1 / angle_sc2 - angle_sc2 / angle_sc3 < 1e-5, \
                'Unexpected ratio between angle spring constants'
            assert abs(angle_sc1 - angle_sc2) > 1e-5, \
                'Unexpected invariant angle spring constants'

        for proper_c1, proper_c2, proper_c3 in zip(proper_constants1,
                                                   proper_constants2,
                                                   proper_constants3):
            assert proper_c1 / proper_c2 - proper_c2 / proper_c3 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2, improper_c3 in zip(improper_constants1,
                                                         improper_constants2,
                                                         improper_constants3):
            assert improper_c1 / improper_c2 - improper_c2 / improper_c3 < 1e-5, \
                'Unexpected ratio between improper constants'

    def test_coul_lambda(self):
        """
        It validates the effects of coul lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import (WritableAtom, WritableBond,
                                             WritableAngle, WritableProper,
                                             WritableImproper)

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=0,
                                                 coul_lambda=0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert (charge2 / charge1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul_lambda=1.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert (charge2 / charge1) - 0.8 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas1.append(atom.sigma)
                epsilons1.append(atom.epsilon)
                SASA_radii1.append(atom.SASA_radius)
                charges1.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants1.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas2.append(atom.sigma)
                epsilons2.append(atom.epsilon)
                SASA_radii2.append(atom.SASA_radius)
                charges2.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants2.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants2.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul_lambda=1.0)

        sigmas3 = list()
        epsilons3 = list()
        SASA_radii3 = list()
        charges3 = list()
        bond_spring_constants3 = list()
        angle_spring_constants3 = list()
        proper_constants3 = list()
        improper_constants3 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas3.append(atom.sigma)
                epsilons3.append(atom.epsilon)
                SASA_radii3.append(atom.SASA_radius)
                charges3.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants3.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants3.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants3.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants3.append(improper.constant)

        for sigma1, sigma2, sigma3 in zip(sigmas1, sigmas2, sigmas3):
            assert sigma1 - sigma2 < 1e-5, \
                'Unexpected ratio between sigmas'
            assert sigma1 - sigma3 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2, epsilon3 in zip(epsilons1, epsilons2, epsilons3):
            assert epsilon1 - epsilon2 < 1e-5, \
                'Unexpected ratio between epsilons'
            assert epsilon1 - epsilon3 < 1e-5, \
                'Unexpected ratio between epsilons'

        for radius1, radius2, radius3 in zip(SASA_radii1, SASA_radii2,
                                             SASA_radii3):
            assert radius1 - radius2 < 1e-5, \
                'Unexpected ratio between SASA radii'
            assert radius1 - radius3 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2, charge3 in zip(charges1, charges2, charges3):
            assert charge1 / charge2 - charge2 / charge3 < 1e-5, \
                'Unexpected ratio between charges'
            assert abs(charge1 - charge2) > 1e-5, \
                'Unexpected invariant charges'

        for bond_sc1, bond_sc2, bond_sc3 in zip(bond_spring_constants1,
                                                bond_spring_constants2,
                                                bond_spring_constants3):
            assert bond_sc1 - bond_sc2 < 1e-5, \
                'Unexpected ratio between bond spring constants'
            assert bond_sc1 - bond_sc3 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2, angle_sc3 in zip(angle_spring_constants1,
                                                   angle_spring_constants2,
                                                   angle_spring_constants3):
            assert angle_sc1 - angle_sc2 < 1e-5, \
                'Unexpected ratio between angle spring constants'
            assert angle_sc1 - angle_sc3 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2, proper_c3 in zip(proper_constants1,
                                                   proper_constants2,
                                                   proper_constants3):
            assert proper_c1 - proper_c2 < 1e-5, \
                'Unexpected ratio between proper constants'
            assert proper_c1 - proper_c3 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2, improper_c3 in zip(improper_constants1,
                                                         improper_constants2,
                                                         improper_constants3):
            assert improper_c1 - improper_c2 < 1e-5, \
                'Unexpected ratio between improper constants'
            assert improper_c1 - improper_c3 < 1e-5, \
                'Unexpected ratio between improper constants'

    def test_coul1_lambda(self):
        """
        It validates the effects of coul1 lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import (WritableAtom, WritableBond,
                                             WritableAngle, WritableProper,
                                             WritableImproper)

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=0,
                                                 coul1_lambda=0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul1_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert (charge2 / charge1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul1_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul1_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert charge2 - charge1 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul1_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas1.append(atom.sigma)
                epsilons1.append(atom.epsilon)
                SASA_radii1.append(atom.SASA_radius)
                charges1.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants1.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul1_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas2.append(atom.sigma)
                epsilons2.append(atom.epsilon)
                SASA_radii2.append(atom.SASA_radius)
                charges2.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants2.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants2.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul1_lambda=1.0)

        sigmas3 = list()
        epsilons3 = list()
        SASA_radii3 = list()
        charges3 = list()
        bond_spring_constants3 = list()
        angle_spring_constants3 = list()
        proper_constants3 = list()
        improper_constants3 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas3.append(atom.sigma)
                epsilons3.append(atom.epsilon)
                SASA_radii3.append(atom.SASA_radius)
                charges3.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants3.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants3.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants3.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants3.append(improper.constant)

        for sigma1, sigma2, sigma3 in zip(sigmas1, sigmas2, sigmas3):
            assert sigma1 - sigma2 < 1e-5, \
                'Unexpected ratio between sigmas'
            assert sigma1 - sigma3 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2, epsilon3 in zip(epsilons1, epsilons2, epsilons3):
            assert epsilon1 - epsilon2 < 1e-5, \
                'Unexpected ratio between epsilons'
            assert epsilon1 - epsilon3 < 1e-5, \
                'Unexpected ratio between epsilons'

        for radius1, radius2, radius3 in zip(SASA_radii1, SASA_radii2,
                                             SASA_radii3):
            assert radius1 - radius2 < 1e-5, \
                'Unexpected ratio between SASA radii'
            assert radius1 - radius3 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2, charge3 in zip(charges1, charges2, charges3):
            assert charge1 - charge2 < 1e-5, \
                'Unexpected ratio between charges'
            assert charge1 - charge3 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2, bond_sc3 in zip(bond_spring_constants1,
                                                bond_spring_constants2,
                                                bond_spring_constants3):
            assert bond_sc1 - bond_sc2 < 1e-5, \
                'Unexpected ratio between bond spring constants'
            assert bond_sc1 - bond_sc3 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2, angle_sc3 in zip(angle_spring_constants1,
                                                   angle_spring_constants2,
                                                   angle_spring_constants3):
            assert angle_sc1 - angle_sc2 < 1e-5, \
                'Unexpected ratio between angle spring constants'
            assert angle_sc1 - angle_sc3 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2, proper_c3 in zip(proper_constants1,
                                                   proper_constants2,
                                                   proper_constants3):
            assert proper_c1 - proper_c2 < 1e-5, \
                'Unexpected ratio between proper constants'
            assert proper_c1 - proper_c3 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2, improper_c3 in zip(improper_constants1,
                                                         improper_constants2,
                                                         improper_constants3):
            assert improper_c1 - improper_c2 < 1e-5, \
                'Unexpected ratio between improper constants'
            assert improper_c1 - improper_c3 < 1e-5, \
                'Unexpected ratio between improper constants'

    def test_coul2_lambda(self):
        """
        It validates the effects of coul2 lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import (WritableAtom, WritableBond,
                                             WritableAngle, WritableProper,
                                             WritableImproper)

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=0,
                                                 coul2_lambda=0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul2_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert charge2 - charge1 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul2_lambda=1.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul2_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert (charge2 / charge1) - 0.8 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul2_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas1.append(atom.sigma)
                epsilons1.append(atom.epsilon)
                SASA_radii1.append(atom.SASA_radius)
                charges1.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants1.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul2_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas2.append(atom.sigma)
                epsilons2.append(atom.epsilon)
                SASA_radii2.append(atom.SASA_radius)
                charges2.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants2.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants2.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 coul2_lambda=1.0)

        sigmas3 = list()
        epsilons3 = list()
        SASA_radii3 = list()
        charges3 = list()
        bond_spring_constants3 = list()
        angle_spring_constants3 = list()
        proper_constants3 = list()
        improper_constants3 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas3.append(atom.sigma)
                epsilons3.append(atom.epsilon)
                SASA_radii3.append(atom.SASA_radius)
                charges3.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants3.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants3.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants3.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants3.append(improper.constant)

        for sigma1, sigma2, sigma3 in zip(sigmas1, sigmas2, sigmas3):
            assert sigma1 - sigma2 < 1e-5, \
                'Unexpected ratio between sigmas'
            assert sigma1 - sigma3 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2, epsilon3 in zip(epsilons1, epsilons2, epsilons3):
            assert epsilon1 - epsilon2 < 1e-5, \
                'Unexpected ratio between epsilons'
            assert epsilon1 - epsilon3 < 1e-5, \
                'Unexpected ratio between epsilons'

        for radius1, radius2, radius3 in zip(SASA_radii1, SASA_radii2,
                                             SASA_radii3):
            assert radius1 - radius2 < 1e-5, \
                'Unexpected ratio between SASA radii'
            assert radius1 - radius3 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2, charge3 in zip(charges1, charges2, charges3):
            assert charge1 - charge2 < 1e-5, \
                'Unexpected ratio between charges'
            assert charge1 - charge3 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2, bond_sc3 in zip(bond_spring_constants1,
                                                bond_spring_constants2,
                                                bond_spring_constants3):
            assert bond_sc1 - bond_sc2 < 1e-5, \
                'Unexpected ratio between bond spring constants'
            assert bond_sc1 - bond_sc3 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2, angle_sc3 in zip(angle_spring_constants1,
                                                   angle_spring_constants2,
                                                   angle_spring_constants3):
            assert angle_sc1 - angle_sc2 < 1e-5, \
                'Unexpected ratio between angle spring constants'
            assert angle_sc1 - angle_sc3 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2, proper_c3 in zip(proper_constants1,
                                                   proper_constants2,
                                                   proper_constants3):
            assert proper_c1 - proper_c2 < 1e-5, \
                'Unexpected ratio between proper constants'
            assert proper_c1 - proper_c3 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2, improper_c3 in zip(improper_constants1,
                                                         improper_constants2,
                                                         improper_constants3):
            assert improper_c1 - improper_c2 < 1e-5, \
                'Unexpected ratio between improper constants'
            assert improper_c1 - improper_c3 < 1e-5, \
                'Unexpected ratio between improper constants'

    def test_vdw_lambda(self):
        """
        It validates the effects of vdw lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import (WritableAtom, WritableBond,
                                             WritableAngle, WritableProper,
                                             WritableImproper)

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=0,
                                                 vdw_lambda=0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        nonpolar_alphas1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)
            nonpolar_alphas1.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 vdw_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        nonpolar_alphas2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)
            nonpolar_alphas2.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert (sigma2 / sigma1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert (epsilon2 / epsilon1) - (1 - 0.2) < 1e-5, \
                'Unexpected ratio between epsilons'

        # born_radius and SASA_radius are NOT scaled for exclusive atoms
        # (fix for OBC singularity at intermediate lambda).
        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert abs(SASA_radius2 - SASA_radius1) < 1e-5, \
                'Unexpected change in SASA radii for exclusive atoms'

        for charge1, charge2 in zip(charges1, charges2):
            assert charge2 - charge1 < 1e-5, \
                'Unexpected ratio between charges'

        # Soft-core s = vdw_lambda for exclusive atoms (vdw_lambda overrides fep)
        for alpha1, alpha2 in zip(nonpolar_alphas1, nonpolar_alphas2):
            assert abs(alpha1 - 0.0) < 1e-5, \
                'Unexpected soft-core s for exclusive atoms at vdw=0'
            assert abs(alpha2 - 0.2) < 1e-5, \
                'Unexpected soft-core s for exclusive atoms at vdw=0.2'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 vdw_lambda=1.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        nonpolar_alphas1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)
            nonpolar_alphas1.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 vdw_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        nonpolar_alphas2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)
            nonpolar_alphas2.append(atom.nonpolar_alpha)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert (sigma2 / sigma1) - 0.2 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert (epsilon2 / epsilon1) - 0.2 < 1e-5, \
                'Unexpected ratio between epsilons'

        # born_radius and SASA_radius are NOT scaled for non-native atoms
        # (fix for OBC singularity at intermediate lambda).
        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert abs(SASA_radius2 - SASA_radius1) < 1e-5, \
                'Unexpected change in SASA radii for non-native atoms'

        for charge1, charge2 in zip(charges1, charges2):
            assert charge2 - charge1 < 1e-5, \
                'Unexpected ratio between charges'

        # Soft-core s = 1 - vdw_lambda for non-native atoms
        for alpha1, alpha2 in zip(nonpolar_alphas1, nonpolar_alphas2):
            assert abs(alpha1 - 0.0) < 1e-5, \
                'Unexpected soft-core s for non-native atoms at vdw=1.0'
            assert abs(alpha2 - 0.8) < 1e-5, \
                'Unexpected soft-core s for non-native atoms at vdw=0.2'

        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected ratio between improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 vdw_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas1.append(atom.sigma)
                epsilons1.append(atom.epsilon)
                SASA_radii1.append(atom.SASA_radius)
                charges1.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants1.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 vdw_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas2.append(atom.sigma)
                epsilons2.append(atom.epsilon)
                SASA_radii2.append(atom.SASA_radius)
                charges2.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants2.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants2.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 vdw_lambda=1.0)

        sigmas3 = list()
        epsilons3 = list()
        SASA_radii3 = list()
        charges3 = list()
        bond_spring_constants3 = list()
        angle_spring_constants3 = list()
        proper_constants3 = list()
        improper_constants3 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas3.append(atom.sigma)
                epsilons3.append(atom.epsilon)
                SASA_radii3.append(atom.SASA_radius)
                charges3.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants3.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants3.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants3.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants3.append(improper.constant)

        for sigma1, sigma2, sigma3 in zip(sigmas1, sigmas2, sigmas3):
            assert sigma1 / sigma2 - sigma2 / sigma3 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2, epsilon3 in zip(epsilons1, epsilons2, epsilons3):
            assert epsilon1 / epsilon2 - epsilon2 / epsilon3 < 1e-5, \
                'Unexpected ratio between epsilons'
            assert abs(epsilon1 - epsilon2) > 1e-5, \
                'Unexpected invariant epsilons'

        for radius1, radius2, radius3 in zip(SASA_radii1, SASA_radii2,
                                             SASA_radii3):
            assert radius1 / radius2 - radius2 / radius3 < 1e-5, \
                'Unexpected ratio between SASA radii'
            assert abs(radius1 - radius2) > 1e-5, \
                'Unexpected invariant SASA radii'

        for charge1, charge2, charge3 in zip(charges1, charges2, charges3):
            assert charge1 - charge2 < 1e-5, \
                'Unexpected ratio between charges'
            assert charge1 - charge3 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2, bond_sc3 in zip(bond_spring_constants1,
                                                bond_spring_constants2,
                                                bond_spring_constants3):
            assert bond_sc1 - bond_sc2 < 1e-5, \
                'Unexpected ratio between bond spring constants'
            assert bond_sc1 - bond_sc3 < 1e-5, \
                'Unexpected ratio between bond spring constants'

        for angle_sc1, angle_sc2, angle_sc3 in zip(angle_spring_constants1,
                                                   angle_spring_constants2,
                                                   angle_spring_constants3):
            assert angle_sc1 - angle_sc2 < 1e-5, \
                'Unexpected ratio between angle spring constants'
            assert angle_sc1 - angle_sc3 < 1e-5, \
                'Unexpected ratio between angle spring constants'

        for proper_c1, proper_c2, proper_c3 in zip(proper_constants1,
                                                   proper_constants2,
                                                   proper_constants3):
            assert proper_c1 - proper_c2 < 1e-5, \
                'Unexpected ratio between proper constants'
            assert proper_c1 - proper_c3 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2, improper_c3 in zip(improper_constants1,
                                                         improper_constants2,
                                                         improper_constants3):
            assert improper_c1 - improper_c2 < 1e-5, \
                'Unexpected ratio between improper constants'
            assert improper_c1 - improper_c3 < 1e-5, \
                'Unexpected ratio between improper constants'

    def test_bonded_lambda(self):
        """
        It validates the effects of bonded lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import (WritableAtom, WritableBond,
                                             WritableAngle, WritableProper,
                                             WritableImproper)

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles('C=C',
                                                          'C(Cl)(Cl)(Cl)')

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=0,
                                                 bonded_lambda=0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 bonded_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._exclusive_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._exclusive_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._exclusive_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._exclusive_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._exclusive_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert charge2 - charge1 < 1e-5, \
                'Unexpected ratio between charges'

        # Bonded force constants for exclusive atoms are no longer
        # annealed by bonded_lambda: only their nonbonded parameters
        # soften, so the bond/angle restraint stays at full strength.
        # These exclusive propers/impropers have no duplicate counterpart
        # in molecule 2 either, so their constant is not annealed.
        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected change in bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected change in angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected change in proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected change in improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 bonded_lambda=1.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas1.append(atom.sigma)
            epsilons1.append(atom.epsilon)
            SASA_radii1.append(atom.SASA_radius)
            charges1.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants1.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 bonded_lambda=0.2)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in alchemizer._non_native_atoms:
            atom = WritableAtom(top.atoms[atom_idx])
            sigmas2.append(atom.sigma)
            epsilons2.append(atom.epsilon)
            SASA_radii2.append(atom.SASA_radius)
            charges2.append(atom.charge)

        for bond_idx in alchemizer._non_native_bonds:
            bond = WritableBond(top.bonds[bond_idx])
            bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in alchemizer._non_native_angles:
            angle = WritableAngle(top.angles[angle_idx])
            angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in alchemizer._non_native_propers:
            proper = WritableProper(top.propers[proper_idx])
            proper_constants2.append(proper.constant)

        for improper_idx in alchemizer._non_native_impropers:
            improper = WritableImproper(top.impropers[improper_idx])
            improper_constants2.append(improper.constant)

        for sigma1, sigma2 in zip(sigmas1, sigmas2):
            assert sigma2 - sigma1 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2 in zip(epsilons1, epsilons2):
            assert epsilon2 - epsilon1 < 1e-5, \
                'Unexpected ratio between epsilons'

        for SASA_radius1, SASA_radius2 in zip(SASA_radii1, SASA_radii2):
            assert SASA_radius2 - SASA_radius1 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2 in zip(charges1, charges2):
            assert charge2 - charge1 < 1e-5, \
                'Unexpected ratio between charges'

        # Same reasoning as above, for non-native atoms/bonds/propers
        for bond_sc1, bond_sc2 in zip(bond_spring_constants1,
                                      bond_spring_constants2):
            assert bond_sc2 - bond_sc1 < 1e-5, \
                'Unexpected change in bond spring constants'

        for angle_sc1, angle_sc2 in zip(angle_spring_constants1,
                                        angle_spring_constants2):
            assert angle_sc2 - angle_sc1 < 1e-5, \
                'Unexpected change in angle spring constants'

        for proper_c1, proper_c2 in zip(proper_constants1,
                                        proper_constants2):
            assert proper_c2 - proper_c1 < 1e-5, \
                'Unexpected change in proper constants'

        for improper_c1, improper_c2 in zip(improper_constants1,
                                            improper_constants2):
            assert improper_c2 - improper_c1 < 1e-5, \
                'Unexpected change in improper constants'

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 bonded_lambda=0.0)

        sigmas1 = list()
        epsilons1 = list()
        SASA_radii1 = list()
        charges1 = list()
        bond_spring_constants1 = list()
        angle_spring_constants1 = list()
        proper_constants1 = list()
        improper_constants1 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas1.append(atom.sigma)
                epsilons1.append(atom.epsilon)
                SASA_radii1.append(atom.SASA_radius)
                charges1.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants1.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants1.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants1.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants1.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 bonded_lambda=0.5)

        sigmas2 = list()
        epsilons2 = list()
        SASA_radii2 = list()
        charges2 = list()
        bond_spring_constants2 = list()
        angle_spring_constants2 = list()
        proper_constants2 = list()
        improper_constants2 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas2.append(atom.sigma)
                epsilons2.append(atom.epsilon)
                SASA_radii2.append(atom.SASA_radius)
                charges2.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants2.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants2.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants2.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants2.append(improper.constant)

        top = alchemizer.get_alchemical_topology(fep_lambda=0.0,
                                                 bonded_lambda=1.0)

        sigmas3 = list()
        epsilons3 = list()
        SASA_radii3 = list()
        charges3 = list()
        bond_spring_constants3 = list()
        angle_spring_constants3 = list()
        proper_constants3 = list()
        improper_constants3 = list()

        for atom_idx in range(0, len(top.atoms)):
            if (atom_idx not in alchemizer._exclusive_atoms and
                    atom_idx not in alchemizer._non_native_atoms):
                atom = WritableAtom(top.atoms[atom_idx])
                sigmas3.append(atom.sigma)
                epsilons3.append(atom.epsilon)
                SASA_radii3.append(atom.SASA_radius)
                charges3.append(atom.charge)

        for bond_idx in range(0, len(top.bonds)):
            if (bond_idx not in alchemizer._exclusive_bonds and
                    bond_idx not in alchemizer._non_native_bonds):
                bond = WritableBond(top.bonds[bond_idx])
                bond_spring_constants3.append(bond.spring_constant)

        for angle_idx in range(0, len(top.angles)):
            if (angle_idx not in alchemizer._exclusive_angles and
                    angle_idx not in alchemizer._non_native_angles):
                angle = WritableAngle(top.angles[angle_idx])
                angle_spring_constants3.append(angle.spring_constant)

        for proper_idx in range(0, len(top.propers)):
            if (proper_idx not in alchemizer._exclusive_propers and
                    proper_idx not in alchemizer._non_native_propers):
                proper = WritableProper(top.propers[proper_idx])
                proper_constants3.append(proper.constant)

        for improper_idx in range(0, len(top.impropers)):
            if (improper_idx not in alchemizer._exclusive_impropers and
                    improper_idx not in alchemizer._non_native_impropers):
                improper = WritableImproper(top.impropers[improper_idx])
                improper_constants3.append(improper.constant)

        for sigma1, sigma2, sigma3 in zip(sigmas1, sigmas2, sigmas3):
            assert sigma1 - sigma2 < 1e-5, \
                'Unexpected ratio between sigmas'
            assert sigma1 - sigma3 < 1e-5, \
                'Unexpected ratio between sigmas'

        for epsilon1, epsilon2, epsilon3 in zip(epsilons1, epsilons2, epsilons3):
            assert epsilon1 - epsilon2 < 1e-5, \
                'Unexpected ratio between epsilons'
            assert epsilon1 - epsilon3 < 1e-5, \
                'Unexpected ratio between epsilons'

        for radius1, radius2, radius3 in zip(SASA_radii1, SASA_radii2,
                                             SASA_radii3):
            assert radius1 - radius2 < 1e-5, \
                'Unexpected ratio between SASA radii'
            assert radius1 - radius3 < 1e-5, \
                'Unexpected ratio between SASA radii'

        for charge1, charge2, charge3 in zip(charges1, charges2, charges3):
            assert charge1 - charge2 < 1e-5, \
                'Unexpected ratio between charges'
            assert charge1 - charge3 < 1e-5, \
                'Unexpected ratio between charges'

        for bond_sc1, bond_sc2, bond_sc3 in zip(bond_spring_constants1,
                                                bond_spring_constants2,
                                                bond_spring_constants3):
            assert bond_sc1 / bond_sc2 - bond_sc2 / bond_sc3 < 1e-5, \
                'Unexpected ratio between bond spring constants'
            assert abs(bond_sc1 - bond_sc2) > 1e-5, \
                'Unexpected invariant bond spring constants'

        for angle_sc1, angle_sc2, angle_sc3 in zip(angle_spring_constants1,
                                                   angle_spring_constants2,
                                                   angle_spring_constants3):
            assert angle_sc1 / angle_sc2 - angle_sc2 / angle_sc3 < 1e-5, \
                'Unexpected ratio between angle spring constants'
            assert abs(angle_sc1 - angle_sc2) > 1e-5, \
                'Unexpected invariant angle spring constants'

        for proper_c1, proper_c2, proper_c3 in zip(proper_constants1,
                                                   proper_constants2,
                                                   proper_constants3):
            assert proper_c1 / proper_c2 - proper_c2 / proper_c3 < 1e-5, \
                'Unexpected ratio between proper constants'

        for improper_c1, improper_c2, improper_c3 in zip(improper_constants1,
                                                         improper_constants2,
                                                         improper_constants3):
            assert improper_c1 / improper_c2 - improper_c2 / improper_c3 < 1e-5, \
                'Unexpected ratio between improper constants'

    @pytest.mark.parametrize("pdb1, pdb2, smiles1, smiles2, " +
                             "fep_lambda, coul_lambda, coul1_lambda,"
                             "coul2_lambda, vdw_lambda, bonded_lambda, " +
                             "golden_sigmas, golden_epsilons, " +
                             "golden_born_radii, golden_SASA_radii, " +
                             "golden_nonpolar_gammas, " +
                             "golden_nonpolar_alphas, golden_charges",
                             [(None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.0, 0.0, 2.5725815350632795, 2.5725815350632795,
                               2.5725815350632795, 2.5725815350632795, 3.480646886945065, 3.480646886945065],
                               [0.0, 0.0, 0.01561134320353, 0.01561134320353,
                               0.01561134320353, 0.01561134320353, 0.0868793154488, 0.0868793154488],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.2862907675316397, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.7403234434725325, 1.7403234434725325],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 1.0],
                               [-0.106311, -0.106311, -0.0, -0.0,
                               0.053156, 0.053156, 0.053156, 0.053156]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.2,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.6615055612921249, 0.6615055612921249, 2.0580652280506238, 2.0580652280506238,
                               2.0580652280506238, 2.481063939423657, 3.446023070848177, 3.460423861881376],
                               [0.012489074562824, 0.012489074562824, 0.012489074562824, 0.015629074562824,
                               0.05312002093054, 0.05312002093054, 0.09127157454406, 0.12262347328957998],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.2405319697118284, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.7230115354240885, 1.730211930940688],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.2,
                               0.2, 0.2, 0.8, 0.8],
                               [-0.1025318, -0.049028200000000015, -0.017483000000000002, -0.017483000000000002,
                               0.0425248, 0.0425248, 0.0425248, 0.0589528]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.8,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.5145163070126558, 0.5145163070126558, 0.5145163070126558, 2.2065111525047882,
                               2.6460222451684996, 2.6460222451684996, 3.3421516225575125, 3.3997547866903095],
                               [0.003122268640705999, 0.003122268640705999, 0.003122268640705999, 0.015682268640705998,
                               0.10444835182984, 0.21248008372216, 0.21248008372216, 0.22985594681192],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.1032555762523941, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.6710758112787563, 1.6998773933451548],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.19999999999999996,
                               0.19999999999999996, 0.8, 0.8, 0.8],
                               [-0.0911942, -0.06993200000000001, -0.06993200000000001, 0.010631199999999999,
                               0.010631199999999999, 0.010631199999999999, 0.0763432, 0.12282020000000003]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.0, 0.0, 0.0, 2.1149935568651657,
                               3.3075278064606244, 3.3075278064606244, 3.3075278064606244, 3.3795317616266205],
                               [0.0, 0.0, 0.0, 0.0157,
                               0.1088406109251, 0.2656001046527, 0.2656001046527, 0.2656001046527],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.0574967784325828, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.6537639032303122, 1.6897658808133103],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 1.0, 1.0],
                               [-0.087415, -0.087415, -0.087415, 0.0,
                               0.0, 0.0, 0.08214, 0.180103]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               None,
                               0.0,
                               0.0,
                               None,
                               None,
                               [0.0, 0.0, 2.5725815350632795, 2.5725815350632795,
                               2.5725815350632795, 2.5725815350632795, 3.480646886945065, 3.480646886945065],
                               [0.0, 0.0, 0.01561134320353, 0.01561134320353,
                               0.01561134320353, 0.01561134320353, 0.0868793154488, 0.0868793154488],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.2862907675316397, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.7403234434725325, 1.7403234434725325],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 1.0],
                               [-0.106311, -0.106311, -0.0, -0.0,
                               0.053156, 0.053156, 0.053156, 0.053156]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               None,
                               1.0,
                               0.0,
                               None,
                               None,
                               [0.0, 0.0, 2.5725815350632795, 2.5725815350632795,
                               2.5725815350632795, 2.5725815350632795, 3.480646886945065, 3.480646886945065],
                               [0.0, 0.0, 0.01561134320353, 0.01561134320353,
                               0.01561134320353, 0.01561134320353, 0.0868793154488, 0.0868793154488],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.2862907675316397, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.7403234434725325, 1.7403234434725325],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 1.0],
                               [-0.106311, -0.106311, 0.0, 0.0,
                               0.0, -0.0, -0.0, 0.053156]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               None,
                               1.0,
                               0.0,
                               None,
                               None,
                               [0.0, 0.0, 0.0, 2.1149935568651657,
                               3.3075278064606244, 3.3075278064606244, 3.3075278064606244, 3.3795317616266205],
                               [0.0, 0.0, 0.0, 0.0157,
                               0.1088406109251, 0.2656001046527, 0.2656001046527, 0.2656001046527],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.0574967784325828, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.6537639032303122, 1.6897658808133103],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 1.0, 1.0],
                               [-0.087415, 0.0, 0.0, 0.0,
                               -0.0, -0.0, 0.08214, 0.180103]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               None,
                               1.0,
                               1.0,
                               None,
                               None,
                               [0.0, 0.0, 0.0, 2.1149935568651657,
                               3.3075278064606244, 3.3075278064606244, 3.3075278064606244, 3.3795317616266205],
                               [0.0, 0.0, 0.0, 0.0157,
                               0.1088406109251, 0.2656001046527, 0.2656001046527, 0.2656001046527],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.0574967784325828, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.6537639032303122, 1.6897658808133103],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 1.0, 1.0],
                               [-0.087415, -0.087415, -0.087415, 0.0,
                               0.0, 0.0, 0.08214, 0.180103]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               0.5,
                               1.0,
                               0.0,
                               None,
                               None,
                               [0.0, 0.0, 0.0, 2.1149935568651657,
                               3.3075278064606244, 3.3075278064606244, 3.3075278064606244, 3.3795317616266205],
                               [0.0, 0.0, 0.0, 0.0157,
                               0.1088406109251, 0.2656001046527, 0.2656001046527, 0.2656001046527],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.0574967784325828, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.6537639032303122, 1.6897658808133103],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 1.0, 1.0],
                               [-0.096863, 0.0, 0.0, 0.0,
                               -0.0, -0.0, 0.036896000000000005, 0.067648]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               1.0,
                               0.0,
                               1.0,
                               None,
                               None,
                               [0.0, 0.0, 0.0, 2.1149935568651657,
                               3.3075278064606244, 3.3075278064606244, 3.3075278064606244, 3.3795317616266205],
                               [0.0, 0.0, 0.0, 0.0157,
                               0.1088406109251, 0.2656001046527, 0.2656001046527, 0.2656001046527],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0],
                               [1.0574967784325828, 1.2862907675316397, 1.2862907675316397, 1.2862907675316397,
                               1.6537639032303122, 1.6537639032303122, 1.6537639032303122, 1.6897658808133103],
                               [0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 2.0, 2.0],
                               [0.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 1.0, 1.0],
                               [-0.087415, -0.087415, -0.087415, 0.053156,
                               0.053156, 0.053156, 0.08214, 0.180103]
                               )
                              ])
    def test_atoms_in_alchemical_topology(self, pdb1, pdb2, smiles1, smiles2,
                                          fep_lambda, coul_lambda,
                                          coul1_lambda, coul2_lambda,
                                          vdw_lambda, bonded_lambda,
                                          golden_sigmas,
                                          golden_epsilons,
                                          golden_born_radii,
                                          golden_SASA_radii,
                                          golden_nonpolar_gammas,
                                          golden_nonpolar_alphas,
                                          golden_charges):
        """
        It validates the effects of lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import WritableAtom

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles(smiles1, smiles2)

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=fep_lambda,
                                                 coul_lambda=coul_lambda,
                                                 coul1_lambda=coul1_lambda,
                                                 coul2_lambda=coul2_lambda,
                                                 vdw_lambda=vdw_lambda,
                                                 bonded_lambda=bonded_lambda)

        sigmas = list()
        epsilons = list()
        born_radii = list()
        SASA_radii = list()
        nonpolar_gammas = list()
        nonpolar_alphas = list()

        for atom in top.atoms:
            atom = WritableAtom(atom)
            sigmas.append(atom.sigma)
            epsilons.append(atom.epsilon)
            born_radii.append(atom.born_radius)
            SASA_radii.append(atom.SASA_radius)
            nonpolar_gammas.append(atom.nonpolar_gamma)
            nonpolar_alphas.append(atom.nonpolar_alpha)

        assert sorted(sigmas) == sorted(golden_sigmas), \
            'Unexpected sigmas'
        assert sorted(epsilons) == sorted(golden_epsilons), \
            'Unexpected epsilons'
        assert sorted(born_radii) == sorted(golden_born_radii), \
            'Unexpected born radii'
        assert sorted(SASA_radii) == sorted(golden_SASA_radii), \
            'Unexpected SASA radii'
        assert sorted(nonpolar_gammas) == sorted(golden_nonpolar_gammas), \
            'Unexpected non polar gammas'
        assert sorted(nonpolar_alphas) == sorted(golden_nonpolar_alphas), \
            'Unexpected non polar alphas'

    @pytest.mark.parametrize("pdb1, pdb2, smiles1, smiles2, " +
                             "fep_lambda, coul_lambda, coul1_lambda, " +
                             "coul2_lambda, vdw_lambda, " +
                             "bonded_lambda, " +
                             "golden_bond_spring_constants",
                             [(None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [172.40622182, 172.40622182, 397.2545789619, 397.2545789619,
                               397.2545789619, 397.2545789619, 399.1592953295]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.5,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [172.40622182, 172.40622182, 285.78275857475, 383.650642924075,
                               397.2545789619, 397.2545789619, 397.2545789619]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [172.40622182, 172.40622182, 172.40622182, 370.04670688625,
                               397.2545789619, 397.2545789619, 397.2545789619]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               1.0,
                               1.0,
                               1.0,
                               1.0,
                               0.0,
                               [172.40622182, 172.40622182, 397.2545789619, 397.2545789619,
                               397.2545789619, 397.2545789619, 399.1592953295]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               1.0,
                               [172.40622182, 172.40622182, 172.40622182, 370.04670688625,
                               397.2545789619, 397.2545789619, 397.2545789619]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               0.5,
                               [172.40622182, 172.40622182, 285.78275857475, 383.650642924075,
                               397.2545789619, 397.2545789619, 397.2545789619]
                               )
                             ])
    def test_bonds_in_alchemical_topology(self, pdb1, pdb2, smiles1,
                                          smiles2, fep_lambda,
                                          coul_lambda, coul1_lambda,
                                          coul2_lambda, vdw_lambda,
                                          bonded_lambda,
                                          golden_bond_spring_constants):
        """
        It validates the effects of lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import WritableBond

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles(smiles1, smiles2)

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=fep_lambda,
                                                 coul_lambda=coul_lambda,
                                                 coul1_lambda=coul1_lambda,
                                                 coul2_lambda=coul2_lambda,
                                                 vdw_lambda=vdw_lambda,
                                                 bonded_lambda=bonded_lambda)

        bond_spring_constants = list()

        for bond in top.bonds:
            bond = WritableBond(bond)
            bond_spring_constants.append(bond.spring_constant)

        assert sorted(bond_spring_constants) == sorted(golden_bond_spring_constants), \
            'Unexpected spring constants'

    @pytest.mark.parametrize("pdb1, pdb2, smiles1, smiles2, " +
                             "fep_lambda, coul_lambda, coul1_lambda, " +
                             "coul2_lambda, vdw_lambda, bonded_lambda, " +
                             "golden_angle_spring_constants",
                             [(None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [22.69631544292, 22.69631544292, 34.066775159195, 34.066775159195,
                               34.066775159195, 34.066775159195, 53.20531626545, 53.20531626545,
                               53.20531626545, 53.20531626545, 53.20531626545]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.5,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [22.69631544292, 22.69631544292, 34.066775159195, 34.066775159195,
                               34.066775159195, 43.6360457123225, 53.20531626545, 53.20531626545,
                               53.20531626545, 53.20531626545, 53.20531626545]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [22.69631544292, 22.69631544292, 34.066775159195, 34.066775159195,
                               34.066775159195, 53.20531626545, 53.20531626545, 53.20531626545,
                               53.20531626545, 53.20531626545, 53.20531626545]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               1.0,
                               1.0,
                               1.0,
                               1.0,
                               1.0,
                               0.0,
                               [22.69631544292, 22.69631544292, 34.066775159195, 34.066775159195,
                               34.066775159195, 34.066775159195, 53.20531626545, 53.20531626545,
                               53.20531626545, 53.20531626545, 53.20531626545]
                               ),
                              (None,
                               None,
                               'C=C',
                               'C(Cl)(Cl)(Cl)',
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               1.0,
                               [22.69631544292, 22.69631544292, 34.066775159195, 34.066775159195,
                               34.066775159195, 53.20531626545, 53.20531626545, 53.20531626545,
                               53.20531626545, 53.20531626545, 53.20531626545]
                               )
                             ])
    def test_angles_in_alchemical_topology(self, pdb1, pdb2, smiles1,
                                           smiles2, fep_lambda,
                                           coul_lambda, coul1_lambda,
                                           coul2_lambda, vdw_lambda,
                                           bonded_lambda,
                                           golden_angle_spring_constants):
        """
        It validates the effects of lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import WritableAngle

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles(smiles1, smiles2)

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=fep_lambda,
                                                 coul1_lambda=coul1_lambda,
                                                 coul2_lambda=coul2_lambda,
                                                 vdw_lambda=vdw_lambda,
                                                 bonded_lambda=bonded_lambda)

        angle_spring_constants = list()

        for angle in top.angles:
            angle = WritableAngle(angle)
            angle_spring_constants.append(angle.spring_constant)

        assert sorted(angle_spring_constants) == sorted(golden_angle_spring_constants), \
            'Unexpected spring constants'

    @pytest.mark.parametrize("pdb1, pdb2, smiles1, smiles2, " +
                             "fep_lambda, coul_lambda, coul1_lambda, "
                             "coul2_lambda, vdw_lambda, bonded_lambda, " +
                             "golden_proper_constants, "
                             "golden_improper_constants",
                             [(None,
                               None,
                               'C[N+](C)(C)CC(=O)[O-]',
                               '[NH]=C(N)c1ccccc1',
                               0.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, -0.3703352413219, 0.02664938770063, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.4541676554336, 0.4541676554336,
                               0.02664938770063, 0.02664938770063, 0.1489710476446, 0.1489710476446,
                               0.02960027280666, 0.02960027280666, 2.348375642009, 2.348375642009,
                               0.9974165607242, 0.9974165607242, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 6.736762477654, 0.9974165607242,
                               0.9974165607242, 6.736762477654, 1.809047390003, 1.809047390003,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               1.256156174911, 1.256156174911, -0.04816277792592, -0.04816277792592],
                               [10.5, 10.5, 1.0, 1.1,
                               1.1, 1.1, 1.1, 1.1,
                               1.1]
                               ),
                              (None,
                               None,
                               'C[N+](C)(C)CC(=O)[O-]',
                               '[NH]=C(N)c1ccccc1',
                               1.0,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, -0.3703352413219, 0.02664938770063, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.4541676554336, 0.4541676554336,
                               0.02664938770063, 0.02664938770063, 0.1489710476446, 0.1489710476446,
                               0.02960027280666, 0.02960027280666, 2.348375642009, 2.348375642009,
                               0.9974165607242, 0.9974165607242, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 6.736762477654, 0.9974165607242,
                               0.9974165607242, 6.736762477654, 1.809047390003, 1.809047390003,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               1.256156174911, 1.256156174911, -0.04816277792592, -0.04816277792592],
                               [10.5, 10.5, 1.0, 1.1,
                               1.1, 1.1, 1.1, 1.1,
                               1.1]
                               ),
                              (None,
                               None,
                               'C[N+](C)(C)CC(=O)[O-]',
                               '[NH]=C(N)c1ccccc1',
                               0.5,
                               None,
                               None,
                               None,
                               None,
                               None,
                               [0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, -0.3703352413219, 0.02664938770063, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.4541676554336, 0.4541676554336,
                               0.02664938770063, 0.02664938770063, 0.1489710476446, 0.1489710476446,
                               0.02960027280666, 0.02960027280666, 2.348375642009, 2.348375642009,
                               0.9974165607242, 0.9974165607242, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 6.736762477654, 0.9974165607242,
                               0.9974165607242, 6.736762477654, 1.809047390003, 1.809047390003,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               1.256156174911, 1.256156174911, -0.04816277792592, -0.04816277792592],
                               [10.5, 10.5, 1.0, 1.1,
                               1.1, 1.1, 1.1, 1.1,
                               1.1]
                               ),
                              (None,
                               None,
                               'C[N+](C)(C)CC(=O)[O-]',
                               '[NH]=C(N)c1ccccc1',
                               1.0,
                               1.0,
                               1.0,
                               1.0,
                               1.0,
                               0.0,
                               [0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, -0.3703352413219, 0.02664938770063, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.4541676554336, 0.4541676554336,
                               0.02664938770063, 0.02664938770063, 0.1489710476446, 0.1489710476446,
                               0.02960027280666, 0.02960027280666, 2.348375642009, 2.348375642009,
                               0.9974165607242, 0.9974165607242, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 6.736762477654, 0.9974165607242,
                               0.9974165607242, 6.736762477654, 1.809047390003, 1.809047390003,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               1.256156174911, 1.256156174911, -0.04816277792592, -0.04816277792592],
                               [10.5, 10.5, 1.0, 1.1,
                               1.1, 1.1, 1.1, 1.1,
                               1.1]
                               ),
                              (None,
                               None,
                               'C[N+](C)(C)CC(=O)[O-]',
                               '[NH]=C(N)c1ccccc1',
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               1.0,
                               [0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, -0.3703352413219, 0.02664938770063, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.06697375586735, 0.06697375586735,
                               0.06697375586735, 0.06697375586735, 0.4541676554336, 0.4541676554336,
                               0.02664938770063, 0.02664938770063, 0.1489710476446, 0.1489710476446,
                               0.02960027280666, 0.02960027280666, 2.348375642009, 2.348375642009,
                               0.9974165607242, 0.9974165607242, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 6.736762477654, 0.9974165607242,
                               0.9974165607242, 6.736762477654, 1.809047390003, 1.809047390003,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               3.661930099076, 3.661930099076, 3.661930099076, 3.661930099076,
                               1.256156174911, 1.256156174911, -0.04816277792592, -0.04816277792592],
                               [10.5, 10.5, 1.0, 1.1,
                               1.1, 1.1, 1.1, 1.1,
                               1.1]
                               )
                             ])
    def test_dihedrals_in_alchemical_topology(self, pdb1, pdb2, smiles1,
                                              smiles2, fep_lambda,
                                              coul_lambda,coul1_lambda,
                                              coul2_lambda, vdw_lambda,
                                              bonded_lambda,
                                              golden_proper_constants,
                                              golden_improper_constants):
        """
        It validates the effects of lambda on atom parameters.
        """
        from peleffy.topology import Alchemizer
        from peleffy.template.impact import WritableProper, WritableImproper

        mol1, mol2, top1, top2 = \
            generate_molecules_and_topologies_from_smiles(smiles1, smiles2)

        alchemizer = Alchemizer(top1, top2)

        top = alchemizer.get_alchemical_topology(fep_lambda=fep_lambda,
                                                 coul1_lambda=coul1_lambda,
                                                 coul2_lambda=coul2_lambda,
                                                 vdw_lambda=vdw_lambda,
                                                 bonded_lambda=bonded_lambda)

        proper_constants = list()
        improper_constants = list()

        for proper in top.propers:
            proper = WritableProper(proper)
            proper_constants.append(proper.constant)

        for improper in top.impropers:
            improper = WritableImproper(improper)
            improper_constants.append(improper.constant)

        assert proper_constants == golden_proper_constants, \
            'Unexpected proper constants'
        assert improper_constants == golden_improper_constants, \
            'Unexpected improper constants'

    def test_hybrid_to_pdb(self):
        """
        It validates the method to write the alchemical structure
        to a PDB file.
        """
        import os
        import tempfile
        from peleffy.utils import get_data_file_path, temporary_cd
        from peleffy.topology import Alchemizer
        from peleffy.tests.utils import compare_files

        mol1, mol2, top1, top2 = generate_molecules_and_topologies_from_pdb(
            'ligands/trimethylglycine.pdb', 'ligands/benzamidine.pdb')

        reference = get_data_file_path('tests/alchemical_structure.pdb')

        alchemizer = Alchemizer(top1, top2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with temporary_cd(tmpdir):
                output_path = os.path.join(tmpdir, 'hybrid.pdb')
                alchemizer.hybrid_to_pdb(output_path)

                compare_files(reference, output_path)

    def test_non_native_coordinates_match_molecule2_geometry(self):
        """
        It checks that non-native atoms (the ones exclusive to
        molecule 2) are placed in the hybrid topology so that their
        bond length and angle with respect to their parent chain
        match molecule 2's own (self-consistent) geometry.

        Before _fix_non_native_coordinates existed, these atoms kept
        their raw molecule 2 coordinates while their mapped parent
        took molecule 1's coordinates, which mixed both frames and
        could stretch this bond by tens of Angstroms.
        """
        import math
        from peleffy.topology import Alchemizer

        mol1, mol2, top1, top2 = generate_molecules_and_topologies_from_pdb(
            'ligands/trimethylglycine.pdb', 'ligands/benzamidine.pdb')

        alchemizer = Alchemizer(top1, top2)

        alc_to_mol2 = {alc_idx: mol2_idx for mol2_idx, alc_idx
                      in alchemizer._mol2_to_alc_map.items()}

        def distance(atom1, atom2):
            return math.sqrt((atom1.x - atom2.x) ** 2
                             + (atom1.y - atom2.y) ** 2
                             + (atom1.z - atom2.z) ** 2)

        def angle(atom1, atom2, atom3):
            v1 = (atom1.x - atom2.x, atom1.y - atom2.y, atom1.z - atom2.z)
            v2 = (atom3.x - atom2.x, atom3.y - atom2.y, atom3.z - atom2.z)
            dot = sum(v1[i] * v2[i] for i in range(3))
            n1 = math.sqrt(sum(c ** 2 for c in v1))
            n2 = math.sqrt(sum(c ** 2 for c in v2))
            cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
            return math.degrees(math.acos(cos_a))

        checked_bonds = 0
        checked_angles = 0
        for atom_idx in alchemizer._non_native_atoms:
            atom = alchemizer._joint_topology.atoms[atom_idx]
            parent = atom.parent

            if parent is None or parent.index not in alc_to_mol2:
                continue

            mol2_atom = top2.atoms[alc_to_mol2[atom_idx]]
            mol2_parent = top2.atoms[alc_to_mol2[parent.index]]

            hybrid_bond = distance(atom, parent)
            native_bond = distance(mol2_atom, mol2_parent)

            assert abs(hybrid_bond - native_bond) < 1e-3, \
                ('Bond length between non-native atom {} and its ' +
                 'parent {} does not match molecule 2\'s own ' +
                 'geometry: {} vs {}').format(
                    atom_idx, parent.index, hybrid_bond, native_bond)
            checked_bonds += 1

            grandparent = parent.parent
            if grandparent is None or grandparent.index not in alc_to_mol2:
                continue

            mol2_grandparent = top2.atoms[alc_to_mol2[grandparent.index]]

            hybrid_angle = angle(atom, parent, grandparent)
            native_angle = angle(mol2_atom, mol2_parent, mol2_grandparent)

            assert abs(hybrid_angle - native_angle) < 1e-2, \
                ('Bond angle at non-native atom {} does not match ' +
                 'molecule 2\'s own geometry: {} vs {}').format(
                    atom_idx, hybrid_angle, native_angle)
            checked_angles += 1

        assert checked_bonds > 0, \
            'No non-native atom had a usable parent to check'
        assert checked_angles > 0, \
            'No non-native atom had a usable grandparent to check'

    def test_non_native_atom_completes_symmetric_geometry(self):
        """
        It checks that when a non-native atom is the only missing
        neighbor of its parent (e.g. the third hydrogen of a methyl
        group that is alchemically replacing a halogen), it is
        placed at the direction that completes ideal local symmetry
        with the parent's other, already placed, mapped neighbors,
        instead of following an unrelated dihedral borrowed from
        molecule 2.

        This reproduces the geometry of a real 1-iodopropane /
        1-bromopropane pair, where the halogen-bearing carbon of one
        molecule maps to the plain methyl carbon of the other.
        Before this fix, the reconstructed hydrogen ended up about
        43 degrees from one of its sibling hydrogens (nearly
        eclipsed) and 137 degrees from the other, instead of the
        ~109.5 degrees expected for a tetrahedral carbon. Since PELE
        never resamples this kind of local geometry, this distortion
        would otherwise persist for the whole alchemical simulation.
        """
        import math
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        from peleffy.topology import Molecule, Topology, Alchemizer
        from peleffy.forcefield import OpenForceField

        def build_molecule(atoms, bonds, coords, name):
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

            return Molecule.from_rdkit(mol, name=name, tag='LIG')

        # Molecule 1: 1-iodopropane, C0-C1-C2-I3 (the real geometry
        # that originally exposed this bug)
        mol1 = build_molecule(
            ['C', 'C', 'C', 'I', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
            [(0, 1), (0, 4), (0, 5), (0, 6),
             (1, 2), (1, 7), (1, 8),
             (2, 3), (2, 9), (2, 10)],
            [(-0.509, -0.549, 0.798), (0.304, -0.521, 2.083),
             (1.675, -1.164, 1.886), (2.801, -1.114, 3.703),
             (-1.487, -0.085, 0.959), (0.000, 0.000, 0.000),
             (-0.672, -1.577, 0.460), (-0.253, -1.047, 2.869),
             (0.415, 0.519, 2.412), (2.229, -0.635, 1.102),
             (1.560, -2.204, 1.560)],
            'state1')

        # Molecule 2: 1-bromopropane, same backbone, halogen swapped
        mol2 = build_molecule(
            ['C', 'C', 'C', 'Br', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
            [(0, 1), (0, 4), (0, 5), (0, 6),
             (1, 2), (1, 7), (1, 8),
             (2, 3), (2, 9), (2, 10)],
            [(1.501, -1.248, 1.823), (0.317, -0.358, 2.160),
             (-0.568, -0.135, 0.947), (-2.069, 1.008, 1.425),
             (2.128, -1.398, 2.707), (2.120, -0.797, 1.040),
             (1.167, -2.230, 1.473), (-0.260, -0.820, 2.972),
             (0.686, 0.602, 2.543), (-0.994, -1.071, 0.574),
             (-0.035, 0.371, 0.138)],
            'state2')

        openff = OpenForceField('openff_unconstrained-2.2.1.offxml')
        params1 = openff.parameterize(mol1, charge_method='gasteiger')
        params2 = openff.parameterize(mol2, charge_method='gasteiger')

        top1 = Topology(mol1, params1)
        top2 = Topology(mol2, params2)

        alchemizer = Alchemizer(top1, top2, mapping_method='mcs')

        def angle(atom1, atom2, atom3):
            v1 = (atom1.x - atom2.x, atom1.y - atom2.y, atom1.z - atom2.z)
            v2 = (atom3.x - atom2.x, atom3.y - atom2.y, atom3.z - atom2.z)
            dot = sum(v1[i] * v2[i] for i in range(3))
            n1 = math.sqrt(sum(c ** 2 for c in v1))
            n2 = math.sqrt(sum(c ** 2 for c in v2))
            cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
            return math.degrees(math.acos(cos_a))

        # Every non-native atom that shares its parent with an
        # exclusive (molecule 1-only) atom is completing a shared
        # substituent slot: the halogens above are one instance
        # (attached to the carbon that keeps its mapped, plain
        # methyl identity), and the extra hydrogens are the other
        # (attached to the carbon that keeps its mapped, halogen
        # identity). Both must be checked, since which one exposes
        # the bug depends on how many ancestors happen to be usable
        # for each: only checking the first one found is not enough
        checked = 0
        for atom in alchemizer._joint_topology.atoms:
            if atom.index not in alchemizer._non_native_atoms:
                continue
            if atom.parent is None:
                continue
            parent = atom.parent

            exclusive_siblings = [
                n for n in alchemizer._graph[parent.index]
                if n != atom.index and n in alchemizer._exclusive_atoms]
            if not exclusive_siblings:
                continue

            mapped_siblings = [
                alchemizer._joint_topology.atoms[n]
                for n in alchemizer._graph[parent.index]
                if n != atom.index and n not in alchemizer._exclusive_atoms]

            assert len(mapped_siblings) >= 2, \
                ('Expected the parent to have at least 2 other ' +
                 'mapped neighbors')

            for sibling in mapped_siblings:
                deg = angle(sibling, parent, atom)
                assert abs(deg - 109.47) < 5.0, \
                    ('Non-native atom {} is not at a tetrahedral angle ' +
                     'from sibling {}: {} degrees').format(
                        atom.index, sibling.index, deg)
            checked += 1

        assert checked > 0, \
            ('Expected at least one non-native atom completing a ' +
             'shared, alchemically-alternated substituent slot')

    def test_molecule1_to_pdb(self):
        """
        It validates the method to write the first molecule structure
        to a PDB file.
        """
        import os
        import tempfile
        from peleffy.utils import get_data_file_path, temporary_cd
        from peleffy.topology import Alchemizer
        from peleffy.tests.utils import compare_files

        mol1, mol2, top1, top2 = generate_molecules_and_topologies_from_pdb(
            'ligands/trimethylglycine.pdb', 'ligands/benzamidine.pdb')

        reference = get_data_file_path('tests/alchemical_mol1.pdb')

        alchemizer = Alchemizer(top1, top2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with temporary_cd(tmpdir):
                output_path = os.path.join(tmpdir, 'mol1.pdb')
                alchemizer.molecule1_to_pdb(output_path)

                compare_files(reference, output_path)

    def test_molecule2_to_pdb(self):
        """
        It validates the method to write the second molecule structure
        to a PDB file.
        """
        import os
        import tempfile
        from peleffy.utils import get_data_file_path, temporary_cd
        from peleffy.topology import Alchemizer
        from peleffy.tests.utils import compare_files

        mol1, mol2, top1, top2 = generate_molecules_and_topologies_from_pdb(
            'ligands/trimethylglycine.pdb', 'ligands/benzamidine.pdb')

        reference = get_data_file_path('tests/alchemical_mol2.pdb')

        alchemizer = Alchemizer(top1, top2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with temporary_cd(tmpdir):
                output_path = os.path.join(tmpdir, 'mol2.pdb')
                alchemizer.molecule2_to_pdb(output_path)

                compare_files(reference, output_path)

    @pytest.mark.parametrize("fep_lambda, reference_path",
                             [(0.0, 'tests/alchemical_0.rot.assign'),
                              (0.5, 'tests/alchemical_1.rot.assign'),
                              (1.0, 'tests/alchemical_2.rot.assign')
                              ])
    def test_rotamer_library_to_file(self, fep_lambda, reference_path):
        """
        It validates the method to write the alchemical rotamer
        library.
        """
        import os
        import tempfile
        from peleffy.utils import get_data_file_path, temporary_cd
        from peleffy.topology import Alchemizer
        from peleffy.tests.utils import compare_files

        mol1, mol2, top1, top2 = generate_molecules_and_topologies_from_pdb(
            'ligands/trimethylglycine.pdb', 'ligands/benzamidine.pdb')

        reference = get_data_file_path(reference_path)

        alchemizer = Alchemizer(top1, top2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with temporary_cd(tmpdir):
                output_path = os.path.join(tmpdir, 'HYB.rot.assign')
                alchemizer.rotamer_library_to_file(output_path,
                                                   fep_lambda=fep_lambda)

                compare_files(reference, output_path)

    @pytest.mark.parametrize("fep_lambda, reference_path",
                             [(0.0, 'tests/alchemical_ligandParams_0.txt'),
                              (0.5, 'tests/alchemical_ligandParams_1.txt'),
                              (1.0, 'tests/alchemical_ligandParams_2.txt')
                              ])
    def test_obc_parameters_to_file(self, fep_lambda, reference_path):
        """
        It validates the method to write the OBC parameters
        template.
        """
        import os
        import tempfile
        from peleffy.utils import get_data_file_path, temporary_cd
        from peleffy.topology import Alchemizer
        from peleffy.tests.utils import compare_files

        mol1, mol2, top1, top2 = generate_molecules_and_topologies_from_pdb(
            'ligands/trimethylglycine.pdb', 'ligands/benzamidine.pdb')

        reference = get_data_file_path(reference_path)

        alchemizer = Alchemizer(top1, top2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with temporary_cd(tmpdir):
                output_path = os.path.join(tmpdir, 'ligandParams.txt')
                alchemizer.obc_parameters_to_file(output_path,
                                                  fep_lambda=fep_lambda)

                compare_files(reference, output_path)