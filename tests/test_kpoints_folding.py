#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:00:27 2026

@author: badal.mondal
"""

import numpy as np
import pickle
from pathlib import Path
import banduppy


# Path to test data
SimulationParentFolder = Path(__file__).parents[0] / "data"


class KpointsFoldingBase:
    """
    Base helper class (NOT collected as test class).
    """

    def generate_sc_kpoints_testing(self): 
        print("- Generating SC Kpoints...")

        band_unfold = banduppy.Unfolding(supercell=self.super_cell_size, 
                                         print_log=None)

        return band_unfold.generate_SC_Kpts_from_pc_k_path(pathPBZ = self.PC_BZ_path,
                                                           nk = self.npoints_per_path_seg,
                                                           labels = self.special_k_points,
                                                           kpts_weights = self.kpts_weights,
                                                           save_kpts = False,
                                                           file_format=self.kpts_file_format)


class TestKpoints_Si8atomUnitcell(KpointsFoldingBase):
    """
    Test class for Si 8-atom unit cell k-point unfolding.
    """

    def setup_method(self):
        """
        pytest will run this BEFORE each test method.
        """

        self.super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] 
        self.PC_BZ_path = [[1/2,1/2,1/2], [0,0,0],[1/2,0,1/2], [5/8,1/4,5/8], None, [3/8,3/8,3/4], [0,0,0]] 
        self.npoints_per_path_seg = (23,27,9,29) 
        self.special_k_points = "LGXUKG"
        self.kpts_weights = 1         

    def test_diff_kpoints_unfolding_vasp(self):
        self.kpts_file_format = 'vasp' # This will generate vasp KPOINTS file format
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        kpointsPBZ_full, _, kpointsSBZ, SBZ_PBZ_kpts_mapping, \
            special_kpoints_pos_labels = self.generate_sc_kpoints_testing()
        print ("- Generating SC Kpoints - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print ("- Reading data from reference saved files...")
        kpoints_file = SimulationParentFolder / 'vasp/Si_8atom/BandUPpyKpointsGen'
        sckp_data = np.genfromtxt(kpoints_file / 'KPOINTS_SC', skip_header=3, comments='!')
        pckp_data = np.genfromtxt(kpoints_file / 'KPOINTS_PC', skip_header=3, comments='!')
        with open(kpoints_file / "KPOINTS_SCPC_map.pkl", "rb") as f:
            sc_pc_kp_map = pickle.load(f)
        with open(kpoints_file / "KPOINTS_SpecialKpoints.pkl", "rb") as f:
            special_kpts = pickle.load(f)
        print ("- Reading reference data files - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Comparing generated data with reference data...")
        np.testing.assert_allclose(kpointsSBZ, sckp_data, rtol=1e-8,atol=1e-5)
        np.testing.assert_allclose(kpointsPBZ_full, pckp_data, rtol=1e-8,atol=1e-5)
        assert special_kpts == special_kpoints_pos_labels
        print("- Comparing generated data with reference data - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print ("** test_diff_kpoints_unfolding_vasp => Congrats! All assertions successfull")