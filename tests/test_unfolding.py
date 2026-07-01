#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:34:20 2026

@author: badal.mondal
"""

import numpy as np
import pickle
import banduppy
from pathlib import Path

# Path to test data
SimulationParentFolder = Path(__file__).parents[0] / "data"
print('SimulationParentFolder:', SimulationParentFolder)

class UnfoldingBase:
    """
    Base helper class (NOT collected as test class).
    """

    def banduppy_unfolding_testing(self, bands, pckpts, sckpts, sc_pc_kp_map): 
        band_unfold = banduppy.Unfolding(supercell=self.super_cell_size, print_log="high")
        

        return band_unfold.Unfold(bands, PBZ_kpts_list_full=pckpts, 
                                  SBZ_kpts_list=sckpts, 
                                  SBZ_PBZ_kpts_map=sc_pc_kp_map,
                                  kline_discontinuity_threshold = 0.1, 
                                  save_unfolded_kpts = {'save2file': False},
                                  save_unfolded_bandstr = {'save2file': False})


class TestBandsUnfolding(UnfoldingBase): 
    """
    Test class to check unfolded bandstructure.
    """
    
    def test_unfolding_Si8atomUnitcell_vasp(self):
        sim_folder = SimulationParentFolder / 'vasp/Si_8atom/VaspSimulations' 
        ref_results_dir = SimulationParentFolder / "vasp/Si_8atom/BandUPpyUnfoldedResults"
        self.super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print ("- Reading band structure data from saved file...")
        unfolded_bandstructure_ref = np.loadtxt(ref_results_dir / 'bandstructure_unfolded.dat')
        kpline_ref = np.loadtxt(ref_results_dir / 'kpoints_unfolded.dat')[:,1]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        kpoints_file = SimulationParentFolder / 'vasp/Si_8atom/BandUPpyKpointsGen'
        sckp = np.genfromtxt(kpoints_file / 'KPOINTS_SC', skip_header=3, comments='!')
        pckp = np.genfromtxt(kpoints_file / 'KPOINTS_PC', skip_header=3, comments='!')
        with open(kpoints_file / "KPOINTS_SCPC_map.pkl", "rb") as f:
            sc_pc_kp_map = pickle.load(f)
        with open(kpoints_file / "KPOINTS_SpecialKpoints.pkl", "rb") as f:
            special_kpoints_pos_labels = pickle.load(f)
        print ("- Reading band structure file - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print ("- Unfolding band structure...")
        bands = banduppy.BandStructure(code="vasp", spinor=False, 
                                       fPOS = sim_folder / "POSCAR",
                                       fWAV = sim_folder / "WAVECAR")
        unfolded_bandstructure, kpline = self.banduppy_unfolding_testing(bands, pckp, sckp,
                                                                         sc_pc_kp_map)
        print ("- Unfolding - done")
       
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Comparing generated data with reference data...")
        np.testing.assert_allclose(unfolded_bandstructure_ref, unfolded_bandstructure, rtol=1e-8,atol=1e-3)
        np.testing.assert_allclose(kpline_ref, kpline, rtol=1e-8,atol=1e-4)
        print("- Comparing generated data with reference data - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print ("- Plotting band structure...")
        plot_unfold = banduppy.Plotting()
        plot_unfold.plot_ebs(kpath_in_angs=kpline,
                              unfolded_bandstructure=unfolded_bandstructure,
                               save_file_name=None, CountFig=None,
                               Ef=5.9786, Emin=-5, Emax=5, pad_energy_scale=0.5,
                               mode="fatband", special_kpoints=special_kpoints_pos_labels,
                               plotSC=True, fatfactor=20, nE=100,smear=0.2,
                               color='red', color_map='viridis', show_colorbar=False)
        print ("- Plotting band structure - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print ("** test_unfolding_Si8atomUnitcell_vasp => Congrats! All assertions successfull")