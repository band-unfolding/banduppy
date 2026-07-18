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
print(f'- BandUPpy version: {banduppy.__version__}')

#==============================================================================
# Path to test data
SimulationParentFolder = Path(__file__).parents[0] / "data"
print(f'- Parent test data folder: {SimulationParentFolder}')

#==============================================================================
class UnfoldingBase:
    """
    Base helper class (NOT collected as test class).
    """

    def banduppy_unfolding_testing(self): 
        band_unfold = banduppy.Unfolding(supercell=self.super_cell_size, print_log="high")
                   
        return band_unfold.Unfold(ab_initio_code=self.ab_initio_code, 
                                  vasp_keywards=self.vasp_keywards,
                                  unfold_kpts_in_batch=self.unfold_kpts_in_batch, 
                                  kpt_batch_size=10,
                                  only_unfold_for_kpts_idxs=self.only_unfold_for_kpts_idxs,
                                  only_unfold_band_idx=self.only_unfold_band_idx,
                                  zero_weight_kp=self.zero_weight_kp,
                                  PBZ_kpts_list_full=self.pckpts, 
                                  SBZ_kpts_list=self.sckpts, 
                                  SBZ_PBZ_kpts_map=self.sc_pc_kp_map,
                                  kline_discontinuity_threshold = 0.1, 
                                  save_unfolded_kpts = {'save2file': False},
                                  save_unfolded_bandstr = {'save2file': False})

#==============================================================================
class TestBandsUnfolding(UnfoldingBase): 
    """
    Test class to check unfolded bandstructure.
    """
    
    def test_unfolding_SiGe8atomUnitcell(self):
        vasp_files_path = 'vasp/SiGe_8atom/'
        sim_folder = SimulationParentFolder / f'{vasp_files_path}/VaspSimulations' 
        ref_results_dir = SimulationParentFolder / f'{vasp_files_path}/BandUPpyUnfoldedResults'
        self.super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Reading band structure data from saved file...")
        unfolded_bandstructure_ref = np.loadtxt(ref_results_dir / 'bandstructure_unfolded.dat')
        kpline_ref = np.loadtxt(ref_results_dir / 'kpoints_unfolded.dat')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        kpoints_file = SimulationParentFolder / f'{vasp_files_path}/BandUPpyKpointsGen'
        self.sckpts = np.genfromtxt(kpoints_file / 'KPOINTS_SC', skip_header=3, comments='#')
        self.pckpts = np.genfromtxt(kpoints_file / 'KPOINTS_PC', skip_header=3, comments='#')
        with open(kpoints_file / "KPOINTS_SCPC_map.pkl", "rb") as f:
            self.sc_pc_kp_map = pickle.load(f)
        with open(kpoints_file / "KPOINTS_SpecialKpoints.pkl", "rb") as f:
            special_kpoints_pos_labels = pickle.load(f)
        print("- Reading band structure file - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Unfolding band structure...")
        self.ab_initio_code = 'vasp'
        self.vasp_keywards = {'poscar_file_path': f'{sim_folder}/POSCAR', 
                              'wavecar_file_path': f'{sim_folder}/WAVECAR', 
                              'vasprunxml_file_path': f'{sim_folder}/vasprun.xml',
                              'is_spin_nondegenrate': False, 
                              'unfold_spin_channel': 'up'}
        self.zero_weight_kp = False
        self.only_unfold_for_kpts_idxs = None
        self.only_unfold_band_idx = (None, None)
        
        # Loading full bandstructure in memory for unfolding 
        # => Dependeing on avaiable RAM, get memory error when wave function file is large ~10 GB
        self.unfold_kpts_in_batch = False
        unfolded_bandstructure, kpline = self.banduppy_unfolding_testing()
        print("- Unfolding that reads full bandstructure - done")
        # Loop over loading one k-point WF, unfold it, delete WF => solves memory problem
        print ("- Unfolding that reads bandstructure kpoints at a time (memory safe)...")
        self.unfold_kpts_in_batch = True
        unfolded_bandstructure_mem, kpline_mem = self.banduppy_unfolding_testing()
        print("- Unfolding that reads bandstructure kpoints at a time (memory safe) - Done")
        print("- Unfolding - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Comparing generated data with reference data...")
        np.testing.assert_allclose(unfolded_bandstructure_ref, unfolded_bandstructure, rtol=1e-8,atol=1e-3)
        np.testing.assert_allclose(kpline_ref, kpline, rtol=1e-8,atol=1e-4)
        print("- Comparing generated data with reference data - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Comparing unfolded bandstructure with memory efficient method...")
        np.testing.assert_allclose(unfolded_bandstructure_mem, unfolded_bandstructure, rtol=1e-8,atol=1e-3)
        np.testing.assert_allclose(kpline_mem, kpline, rtol=1e-8,atol=1e-4)
        print("- Comparing unfolded bandstructure with memory efficient method - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Plotting band structure...")
        unfolded_bandstructure, kpline = self.banduppy_unfolding_testing()
        plot_unfold = banduppy.Plotting()
        _ = plot_unfold.plot_ebs(unfolded_kpoints =ref_results_dir / 'kpoints_unfolded.dat', 
                                 unfolded_bandstructure =ref_results_dir / 'bandstructure_unfolded.dat', 
                                 #unfolded_kpoints=kpline,
                                 #unfolded_bandstructure=unfolded_bandstructure,
                                 save_file_name=None, CountFig=None,
                                 Ef=0.0, Emin=-5, Emax=5, pad_energy_scale=0.5,
                                 mode="fatband", special_kpoints=special_kpoints_pos_labels,
                                 plotSC=True, fatfactor=20, nE=100,smear=0.2,
                                 color='red', color_map='viridis', 
                                 show_colorbar=False, show_plot=False)
        print("- Plotting band structure - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Plotting band structure with only part of the K-points from map file are unfolded...")
        Kkmap_tmp = self.sc_pc_kp_map.copy()
        self.sc_pc_kp_map = {k:Kkmap_tmp[k] for k in range(5,10)}
        unfolded_bandstructure, kpline = self.banduppy_unfolding_testing()
        for special_kpts in [special_kpoints_pos_labels, None]:
            plot_unfold = banduppy.Plotting()
            _ = plot_unfold.plot_ebs(unfolded_kpoints=kpline,
                                     unfolded_bandstructure=unfolded_bandstructure,
                                     save_file_name=None, CountFig=None, Emin=-5, 
                                     Ef=None, Emax=5, pad_energy_scale=0.5,
                                     mode="fatband", special_kpoints=special_kpts,
                                     plotSC=True, fatfactor=20, nE=100,smear=0.2,
                                     color='red', color_map='viridis', 
                                     show_colorbar=False, show_plot=False)
        self.sc_pc_kp_map = Kkmap_tmp
        print("- Plotting band structure with only part of the K-points from map file are unfolded - done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("- Unfolding with user specified list of kpoint indices...")
        self.only_unfold_for_kpts_idxs = [5]
        self.unfold_kpts_in_batch = False
        unfolded_bandstructure_kplist, kpline_kplist = self.banduppy_unfolding_testing()
        print("- Unfolding with user specified list of kpoint indices - Done")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("** test_unfolding_Si8atomUnitcell_vasp => Congrats! All assertions successfull")