"""
This module contains the tests to check all available force fields in
offpele.
"""


class TestOpenForceField(object):
    """
    It wraps all tests that check the OpenForceField class.
    """

    def test_name(self):
        """It checks the name assignment."""

        from offpele.forcefield import OpenForceField

        FORCE_FIELD_NAME = 'openff_unconstrained-1.2.1.offxml'

        openff = OpenForceField(FORCE_FIELD_NAME)

        assert openff.name == FORCE_FIELD_NAME, \
            'Unexpected force field name'

    def test_type(self):
        """It checks the type assignment."""

        from offpele.forcefield import OpenForceField

        FORCE_FIELD_NAME = 'openff_unconstrained-1.2.1.offxml'

        openff = OpenForceField(FORCE_FIELD_NAME)

        assert openff.type == 'OpenFF', \
            'Unexpected force field type'

    def test_parameterizer(self):
        """It checks the parameterized method."""

        from offpele.topology import Molecule
        from offpele.forcefield import OpenForceField
        from offpele.utils import get_data_file_path
        from .utils import check_parameters

        FORCE_FIELD_NAME = 'openff_unconstrained-1.2.1.offxml'

        # Load molecule
        molecule = Molecule(get_data_file_path('ligands/methane.pdb'))
        openff = OpenForceField(FORCE_FIELD_NAME)

        # Set force field and obtain parameters
        molecule.set_forcefield(openff)
        molecule.parameterize()

        # Define expected parameters
        expected_nonbonding = [
            [1, 0, 'M', 'OFFT', '_C1_', 0, 3.3996695084235347, 0.1094,
             -0.1088, 0, 1.6998347542117673, 0, 0],
            [2, 1, 'M', 'OFFT', '_H1_', 0, 2.649532787749369, 0.0157,
             0.0267, 0, 1.3247663938746845, 0, 0],
            [3, 1, 'M', 'OFFT', '_H2_', 0, 2.649532787749369, 0.0157,
             0.0267, 0, 1.3247663938746845, 0, 0],
            [4, 1, 'M', 'OFFT', '_H3_', 0, 2.649532787749369, 0.0157,
             0.0267, 0, 1.3247663938746845, 0, 0],
            [5, 1, 'M', 'OFFT', '_H4_', 0, 2.649532787749369, 0.0157,
             0.0267, 0, 1.3247663938746845, 0, 0]]

        expected_bonds = [
            [1, 2, 376.8940758588, 1.094223427522],
            [1, 3, 376.8940758588, 1.094223427522],
            [1, 4, 376.8940758588, 1.094223427522],
            [1, 5, 376.8940758588, 1.094223427522]]

        expected_angles = [
            [2, 1, 3, 33.78875634641, 110.2468561538],
            [2, 1, 4, 33.78875634641, 110.2468561538],
            [2, 1, 5, 33.78875634641, 110.2468561538],
            [3, 1, 4, 33.78875634641, 110.2468561538],
            [3, 1, 5, 33.78875634641, 110.2468561538],
            [4, 1, 5, 33.78875634641, 110.2468561538]]

        # Check it up
        check_parameters(molecule,
                         expected_nonbonding=expected_nonbonding,
                         expected_bonds=expected_bonds,
                         expected_angles=expected_angles)

        # Load molecule
        molecule = Molecule(get_data_file_path('ligands/ethylene.pdb'))
        openff = OpenForceField(FORCE_FIELD_NAME)

        # Set force field and obtain parameters
        molecule.set_forcefield(openff)
        molecule.parameterize()

        # Define expected parameters
        expected_propers = [
            [3, 1, 2, 5, 5.376019778605, -1, 2, 0.0],
            [3, 1, 2, 6, 5.376019778605, -1, 2, 0.0],
            [4, 1, 2, 5, 5.376019778605, -1, 2, 0.0],
            [4, 1, 2, 6, 5.376019778605, -1, 2, 0.0]]

        expected_impropers = [
            [1, 2, 5, 6, 1.1, -1, 2],
            [2, 1, 3, 4, 1.1, -1, 2]]

        # Check it up
        check_parameters(molecule,
                         expected_propers=expected_propers,
                         expected_impropers=expected_impropers)


class TestOPLS2005ForceField(object):
    """
    It wraps all tests that check the OPLS2005ForceField class.
    """

    def test_name(self):
        """It checks the name assignment."""

        from offpele.forcefield import OPLS2005ForceField

        FORCE_FIELD_NAME = 'OPLS2005'

        oplsff = OPLS2005ForceField(FORCE_FIELD_NAME)

        assert oplsff.name == FORCE_FIELD_NAME, \
            'Unexpected force field name'

    def test_type(self):
        """It checks the type assignment."""

        from offpele.forcefield import OPLS2005ForceField

        FORCE_FIELD_NAME = 'OPLS2005'

        oplsff = OPLS2005ForceField(FORCE_FIELD_NAME)

        assert oplsff.type == 'OPLS2005', \
            'Unexpected force field type'

    def test_parameterizer(self):
        """It checks the parameterized method."""

        from offpele.topology import Molecule
        from offpele.forcefield import (OPLS2005ForceField,
                                        OPLS2005ParameterWrapper)
        from offpele.utils import get_data_file_path
        from .utils import check_parameters

        FORCE_FIELD_NAME = 'OPLS2005'

        # Load molecule
        molecule = Molecule(get_data_file_path('ligands/methane.pdb'))
        oplsff = OPLS2005ForceField(FORCE_FIELD_NAME)

        # Set force field and obtain parameters
        ffld_output = get_data_file_path('tests/MET_ffld_output.txt')
        parameters = OPLS2005ParameterWrapper.from_ffld_output(ffld_output)
        molecule.set_forcefield(oplsff)
        molecule._parameters = parameters

        molecule._clean_lists()
        molecule._build_atoms()
        molecule._build_bonds()
        molecule._build_angles()
        molecule._build_propers()
        molecule._build_impropers()
        molecule.graph.set_core()
        molecule.graph.set_parents()

        # Define expected parameters
        expected_nonbonding = [
            [1, 0, 'M', 'CT', '_C1_', 0, 3.5, 0.066, -0.24, 1.975, 1.75,
             0.005, -0.74168571],
            [2, 1, 'M', 'HC', '_H2_', 0, 2.5, 0.03, 0.06, 1.425, 1.25,
             0.00859824, 0.268726247],
            [3, 1, 'M', 'HC', '_H3_', 0, 2.5, 0.03, 0.06, 1.425, 1.25,
             0.00859824, 0.268726247],
            [4, 1, 'M', 'HC', '_H4_', 0, 2.5, 0.03, 0.06, 1.425, 1.25,
             0.00859824, 0.268726247],
            [5, 1, 'M', 'HC', '_H5_', 0, 2.5, 0.03, 0.06, 1.425, 1.25,
             0.00859824, 0.268726247]]

        expected_bonds = [
            [1, 2, 340.0, 1.09],
            [1, 3, 340.0, 1.09],
            [1, 4, 340.0, 1.09],
            [1, 5, 340.0, 1.09]]

        expected_angles = [
            [2, 1, 3, 33.0, 107.8],
            [2, 1, 4, 33.0, 107.8],
            [2, 1, 5, 33.0, 107.8],
            [3, 1, 4, 33.0, 107.8],
            [3, 1, 5, 33.0, 107.8],
            [4, 1, 5, 33.0, 107.8]]

        # Check it up
        check_parameters(molecule,
                         expected_nonbonding=expected_nonbonding,
                         expected_bonds=expected_bonds,
                         expected_angles=expected_angles)

        # Load molecule
        molecule = Molecule(get_data_file_path('ligands/ethylene.pdb'))
        oplsff = OPLS2005ForceField(FORCE_FIELD_NAME)

        # Set force field and obtain parameters
        ffld_output = get_data_file_path('tests/ETL_ffld_output.txt')
        parameters = OPLS2005ParameterWrapper.from_ffld_output(ffld_output)
        molecule.set_forcefield(oplsff)
        molecule._parameters = parameters

        molecule._clean_lists()
        molecule._build_atoms()
        molecule._build_bonds()
        molecule._build_angles()
        molecule._build_propers()
        molecule._build_impropers()

        # Define expected parameters
        expected_propers = [
            [3, 1, 2, 5, 7.0, -1, 2, 0.0],
            [3, 1, 2, 6, 7.0, -1, 2, 0.0],
            [4, 1, 2, 5, 7.0, -1, 2, 0.0],
            [4, 1, 2, 6, 7.0, -1, 2, 0.0]]

        expected_impropers = [
            [3, 4, 1, 2, 15.0, -1, 2],
            [5, 6, 2, 1, 15.0, -1, 2]]

        # Check it up
        check_parameters(molecule,
                         expected_propers=expected_propers,
                         expected_impropers=expected_impropers)

    def test_add_solvent_parameters(self):
        """
        It tests the function that adds the solvent parameters to
        the OPLSParameters collection.
        """

        from simtk import unit
        from offpele.forcefield import OPLS2005ParameterWrapper

        # Using a standard atom type
        params1 = OPLS2005ParameterWrapper(
            {'atom_names': [' C1 ', ' H1 ', ' H2 ', ' H3 ', ' H4 '],
             'atom_types': ['CT', 'HC', 'HC', 'HC', 'HC'],
             'charges': [-0.24, 0.06, 0.06, 0.06, 0.06],
             'sigmas': [3.5, 2.5, 2.5, 2.5, 2.5],
             'epsilons': [0.066, 0.03, 0.03, 0.03, 0.03]})

        # Using a similar atom type
        params2 = OPLS2005ParameterWrapper(
            {'atom_names': [' C1 ', ' H1 ', ' H2 ', ' H3 ', ' H4 '],
             'atom_types': ['C3M', 'HC', 'HC', 'HC', 'HC'],
             'charges': [-0.24, 0.06, 0.06, 0.06, 0.06],
             'sigmas': [3.5, 2.5, 2.5, 2.5, 2.5],
             'epsilons': [0.066, 0.03, 0.03, 0.03, 0.03]})

        # Using a default atom type
        params3 = OPLS2005ParameterWrapper(
            {'atom_names': [' C1 ', ' H1 ', ' H2 ', ' H3 ', ' H4 '],
             'atom_types': ['XX', 'HC', 'HC', 'HC', 'HC'],
             'charges': [-0.24, 0.06, 0.06, 0.06, 0.06],
             'sigmas': [3.5, 2.5, 2.5, 2.5, 2.5],
             'epsilons': [0.066, 0.03, 0.03, 0.03, 0.03]})

        OPLS2005ParameterWrapper._add_solvent_parameters(params1)
        OPLS2005ParameterWrapper._add_solvent_parameters(params2)
        OPLS2005ParameterWrapper._add_solvent_parameters(params3)

        assert params1['SGB_radii'][0] == \
            unit.Quantity(1.975, unit.angstrom), 'Unexpected SGB radius'
        assert params1['vdW_radii'][0] == \
            unit.Quantity(1.750, unit.angstrom), 'Unexpected vdW radius'
        assert params1['gammas'][0] == 0.005000000, 'Unexpected gamma'
        assert params1['alphas'][0] == -0.741685710, 'Unexpected alpha'

        assert params2['SGB_radii'][0] == \
            unit.Quantity(2.002, unit.angstrom), 'Unexpected SGB radius'
        assert params2['vdW_radii'][0] == \
            unit.Quantity(1.775, unit.angstrom), 'Unexpected vdW radius'
        assert params2['gammas'][0] == 0.023028004, 'Unexpected gamma'
        assert params2['alphas'][0] == -0.852763146, 'Unexpected alpha'

        assert params3['SGB_radii'][0] == \
            unit.Quantity(1.500, unit.angstrom), 'Unexpected SGB radius'
        assert params3['vdW_radii'][0] == \
            unit.Quantity(1.250, unit.angstrom), 'Unexpected vdW radius'
        assert params3['gammas'][0] == 0.005000000, 'Unexpected gamma'
        assert params3['alphas'][0] == 0.000000000, 'Unexpected alpha'
