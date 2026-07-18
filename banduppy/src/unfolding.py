import numpy as np
#import warnings
from ..BasicFunctions.general_functions import _SaveData2File, _BasicFunctionsModule, _draw_line_length
from .. import __version__

### ===========================================================================
class _GeneralFnsDefs:
    """
    Some of the general function definitions.
    """
    @classmethod
    def _save_Post_unfolded_PBZ_kpts(cls, unfolded_kpts_dat, save_dir, file_name, 
                                     file_name_suffix, print_information='low'):
        """
        Save unfolded PC-kpoints.

        Parameters
        ----------
        unfolded_kpts_dat : numpy array
            Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]
        save_dir : str or path
            Directory path where to save the file.
        file_name : str
            Name of the file.
        file_name_suffix : str
            Suffix to add to the file name.
        print_information : [None,'low','medium','high'], optional
                Level of printing information. 
                The default is 'low'. If None, nothing is printed.

        Returns
        -------
        None.

        """
        if print_information is not None: 
            print(f"{'='*_draw_line_length}\n- Saving unfolded kpoints to file...")
        header_msg  = f" Unfolded PC k-points from postprocessed wavefunction file [banduppy-{__version__}]\n"
        header_msg += " k-index, k on path (A^-1), k1, k2, k3\n"
        n_cols_data = unfolded_kpts_dat.shape[1]
        kp_ind_max_len = len(str(max(unfolded_kpts_dat[:,0])))
        np_data_fmt = [f'%{kp_ind_max_len}d'] + ['%12.8f']*(n_cols_data-1)
        # Save the sc-kpoints in file
        save_f_name = \
        _SaveData2File._save_2_file(data=unfolded_kpts_dat, 
                                    save_dir=save_dir, file_name=file_name,
                                    file_name_suffix=f'{file_name_suffix}.dat', 
                                    header_txt=header_msg, comments_symbol='#',
                                    np_data_fmt=np_data_fmt,
                                    print_log=bool(print_information))
        if print_information is not None: 
            print(f'-- Filepath: {save_f_name}\n- Done')

    @classmethod
    def _save_Post_unfolded_bandstucture(cls, unfolded_bandstructure, save_dir, file_name, file_name_suffix, 
                                         print_information='low', is_spinor:bool=False):
        """
        Save unfolded effective band structure data.

        Parameters
        ----------
        unfolded_bandstructure : numpy ndarray
            Unfolded effective band structure.
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor]
        save_dir : str or path
            Directory path where to save the file.
        file_name : str
            Name of the file.
        file_name_suffix : str
            Suffix to add to the file name.
        is_spinor : bool, optional
            If the bands in wave function files are non-degenerate (spinor). 
            The default is False.
        print_information : [None,'low','medium','high'], optional
                Level of printing information. 
                The default is 'low'. If None, nothing is printed.

        Returns
        -------
        None.

        """
        if print_information is not None: 
            print(f"{'='*_draw_line_length}\n- Saving unfolded bandstructure to file...")
        header_msg  = f" Unfolded band structure from postprocessed wavefunction file [banduppy-{__version__}]\n"
        header_msg += " k-index, k on path (A^-1), energy (eV), weight " + \
                        ("Sx,Sy,Sz" if is_spinor else "")  +"\n"
        n_cols_data = unfolded_bandstructure.shape[1]
        kp_ind_max_len = len(str(max(unfolded_bandstructure[:,0])))
        np_data_fmt = [f'%{kp_ind_max_len}d'] + ['%12.8f']*(n_cols_data-1)
        # Save the unfolded band structure in file
        save_f_name = \
        _SaveData2File._save_2_file(data=unfolded_bandstructure, 
                                    save_dir=save_dir, file_name=file_name,
                                    file_name_suffix=f'{file_name_suffix}.dat', 
                                    header_txt=header_msg, comments_symbol='#',
                                    np_data_fmt=np_data_fmt,
                                    print_log=bool(print_information)) 
        if print_information is not None: 
            print(f'-- Filepath: {save_f_name}\n- Done')

class _BandUnfolding(_GeneralFnsDefs):
    """
    Band unfolding from supercell to primitive cell band structure (effective band structure).

    """
    
    def __init__(self, supercell, PBZ_kpts_list_full,  
                 SBZ_kpts_list, SBZ_PBZ_kpts_map, 
                 print_info='low'):
        """
        Initializing the unfolding class.

        Parameters
        ----------
        supercell : 3X3 matrix
            Primitive-to-supercell transformation matrix.
        PBZ_kpts_list_full : ndarray of floats
            PC kpoints list (orginal provided k-path, full list).
        SBZ_kpts_list : ndarray of float
            SC kpoints list (unique).
        SBZ_PBZ_kpts_map : dictionary of int/list
            Mapping of SC kpts (K), PC unique kpts (k unique), and PC full kpts (k) indices.
            format: {K index: K -> k index unique: k unique -> k index: k}
            This mapping is used for reverse engineer latter.
        print_info : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """
        self.transformation_matrix = _BasicFunctionsModule._check_transformation_matrix(np.array(supercell))
        self.kpointsPBZ_full = _BasicFunctionsModule._round_2_tolerance(PBZ_kpts_list_full) 
        self.kpointsSBZ = _BasicFunctionsModule._round_2_tolerance(SBZ_kpts_list)
        self.SBZ_PBZ_kpts_index_map = SBZ_PBZ_kpts_map
        self.print_information = print_info
    
    def _gather_generated_ab_calculated_kpts_old(self, wavefns_file_kpts):
        ### THIS FUNCTION CAN BE REMOVED LATER
        # Same function as _perform_unfolding() but older version.
        """
        Collect Kpoints in the generated SC-Kpoints list those only exist in the 
        wavefunction file from ab-initio calculation. 
        
        Note: This way you can do sections of k-paths in ab-initio calculations to avoid
        large calculations. Carefully check the WARNING msgs.
        
        Note: Quick theory:
        If SC reciprocal lattice =  G(k<-K) 
        o K unfold onto k with the unfolding vector G(k<-K). One K can unfold to multiple k.
        o But a given k can fold to only one K.

        Parameters
        ----------
        wavefns_file_kpts : irrep.kpoint.Kpoint
            List of irrep.kpoint.Kpoint from ab-initio calculations.

        Calculates
        -------
        kpSBZcalc Dictionary = 
            {SC-Kpoint index: 
                 unfolded PC-kpoint index: (band energy, Bloch weights).}

        """
        self.kpSBZcalc = {}
        for key, val in self.SBZ_PBZ_kpts_index_map.items(): # Loop over all SC-Kpoints
            found = False
            for KP in wavefns_file_kpts:
                # Check if K exists in wavefunction file K-list
                if KP.k_close_mod1(self.kpointsSBZ[key, :3], prec=1e-6):
                    self.kpSBZcalc[key] = {}
                    for vall in val.values(): # Loop over PC-kpoints
                        for kk in vall:
                            # Calculate weights for the PBZ kpoints from SBZ kpoints on which
                            # it was folded.
                            # If SC reciprocal lattice =  G(k<-K) 
                            # o K unfold onto k with the unfolding vector G(k<-K). One K can unfold to multiple k.
                            # o But a given k can fold to only one K.
                            self.kpSBZcalc[key][kk] = KP.unfold(supercell=self.transformation_matrix, 
                                                                kptPBZ=self.kpointsPBZ_full[kk, :3])
                    found = True
            if not found:
                print("WARNING: SC K-point "+f"{key:>5}"+"  ".join(f"{kpp:12.8f}" for kpp in self.kpointsSBZ[key])+" was not found in the calculated bandstructure.")
                print("- The corresponding following PC-kpoints in the unfolding path will be skipped:")
                for _, vall in val.items(): # kkk: unique PC-kpts indices; vall: list of PC-kpts indices
                    for kk in vall:
                            print(f"\t--- {kk:>5}:" + "  ".join(f"{x:12.8f}" for x in self.kpointsPBZ_full[kk, :3])) 
                            
    def _perform_unfolding(self, wavefns_file_kpts) -> None:
        """
        This function perform unfolding.
        
        Note: It collects Kpoints in the generated SC-Kpoints list those only exist in the 
        wavefunction file from ab-initio calculation. This way you can do sections of 
        k-paths in ab-initio calculations to avoid large calculations. 
        Carefully check the WARNING msgs.
        
        Note: Quick theory:
        If SC reciprocal lattice =  G(k<-K) 
        o K unfold onto k with the unfolding vector G(k<-K). One K can unfold to multiple k.
        o But a given k can fold to only one K.

        Parameters
        ----------
        wavefns_file_kpts : irrep.kpoint.Kpoint
            List of irrep.kpoint.Kpoint from ab-initio calculations.

        Calculates
        -------
        kpSBZcalc Dictionary = 
            {SC-Kpoint index: 
                 unfolded PC-kpoint index: (band energy, Bloch weights).}

        """

        if not self.kpSBZcalc:
            self.tmp_Kk_map = self.SBZ_PBZ_kpts_index_map.copy()
            # Otherwise we wouldn't know on which k-points, the K-point should be unfolded. Mapping info is missing.
            assert len(self.kpointsSBZ) >= len(self.tmp_Kk_map), 'Total no. of SC K-points in supplied SBZ file has to be greater than the no. in K-k map file.'
        
        for KP in wavefns_file_kpts:
            # Check if WAVECAR K exists in SC file K-list
            Kk_match_found = False
            for key, val in self.tmp_Kk_map.items(): # Loop over all SC-Kpoints
                if KP.k_close_mod1(self.kpointsSBZ[key, :3], prec=1e-6):
                    #print(f'K-K match found: {key}, Map dict len:{len(self.tmp_Kk_map)}')
                    self.kpSBZcalc[key] = {}
                    for vall in val.values(): # Loop over PC-kpoints
                        for kk in vall:
                            # Calculate weights for the PBZ kpoints from SBZ kpoints on which
                            # it was folded.
                            # If SC reciprocal lattice =  G(k<-K) 
                            # o K unfold onto k with the unfolding vector G(k<-K). One K can unfold to multiple k.
                            # o But a given k can fold to only one K.
                            self.kpSBZcalc[key][kk] = KP.unfold(supercell=self.transformation_matrix, 
                                                                kptPBZ=self.kpointsPBZ_full[kk, :3])
                    Kk_match_found = True
                    break
            if Kk_match_found:
                del self.tmp_Kk_map[key]
            else:
                kp_coord = 'Wavefunction file contains SC K-point: ['+ ",".join(f"{kpp:12.8f}" for kpp in KP.k) +  ']'
                print(f"WARNING: {kp_coord}. But no PC k-point mapping is found in KPOINTS_SCPC_map file. Ignoring unfolding of this SC K-point.")         
        return 
    
    def _print_warning_if_Kk_map_file_not_exhausted(self) -> None:
        """
        This function prints warning message if there are K-points that are in the
        user specified SC_K-PC_k map file but does not exists in the wave function
        file. 

        Returns
        -------
        None
        """
        for key, val in self.tmp_Kk_map.items():
            kp_coord = f'KPOINTS_SCPC_map file contans SC K-point {key:>5} [' + ",".join(f"{kpp:12.8f}" for kpp in self.kpointsSBZ[key]) + ']'
            print(f"WARNING: {kp_coord}. Can't find this SC K-point in the wavefunction file. ")
            print("- Accordingly, following PC-kpoints in the unfolding path will be skipped:")
            for _, vall in val.items(): # kkk: unique PC-kpts indices; vall: list of PC-kpts indices
                for kk in vall:
                        print(f"\t--- {kk:>5}:" + "  ".join(f"{x:12.8f}" for x in self.kpointsPBZ_full[kk, :3])) 
                        
    def  _split_kp_slices(self, kpt_list):
        if self.kpt_batch_size < 1: 
            print('WARNING: Minimum batch size should be = 1. Re-setting kpt_batch_size=1')
            self.kpt_batch_size = 1
        n_kpts = len(kpt_list)
        if self.kpt_batch_size > n_kpts: 
            print('WARNING: Maximum batch size should be total number of kpoints. Re-setting kpt_batch_size to Max value.')
            self.kpt_batch_size = n_kpts            
        iklist = list(range(n_kpts))
        return np.array_split(iklist, np.arange(self.kpt_batch_size, n_kpts, self.kpt_batch_size)) 
        
    def _unfold_bandstructure(self, bandstructure):  
        """

        Parameters
        ----------
        bandstructure : BandStructure instance from irrep.bandstructure 
            BandStructure instance from irrep.bandstructure class.

        Returns
        -------
        None.

        """            
        # Unfolding part
        if self.print_information == 'high':
            kpt_list = bandstructure.kplist
            print(f'- Total number of unfolded K-points: {len(kpt_list)}')
            print(f"- K-point indices: [{','.join([str(ll) for ll in kpt_list])}]")
            
        self.kpSBZcalc = {}
        if self.unfold_in_batch:
            # Split into slices avoids large Memory requirement
            iklist_split = self._split_kp_slices(bandstructure.kplist)
            if self.reading_gpaw_paw: 
                for iklist in iklist_split:
                    for ik in iklist: 
                        bandstructure.set_kpoint(ik) 
                        bandstructure.set_kpoint_paw(ik)
                    kpoints_slice = [bandstructure.kpoints[ik] for ik in iklist]
                    self._perform_unfolding(kpoints_slice)
                    for ik in iklist: 
                        bandstructure.forget_kpoint(ik)
            else:        
                for iklist in iklist_split:
                    for ik in iklist: 
                        bandstructure.set_kpoint(ik) 
                    # if self.reading_gpaw_paw: # Don't like it, unnecessary loop conditional check for other codes
                    #     for ik in iklist: 
                    #         bandstructure.set_kpoint_paw(ik)
                    kpoints_slice = [bandstructure.kpoints[ik] for ik in iklist]
                    self._perform_unfolding(kpoints_slice)
                    for ik in iklist: 
                        bandstructure.forget_kpoint(ik)
        else:
            self._perform_unfolding(bandstructure.kpoints)
            
        if not self.is_wf_file_contain_only_bandstr_sec:
            self._print_warning_if_Kk_map_file_not_exhausted()
        
        # Collect all unfolded PBZ kpoints, energy and weights from calculated kpSBZcalc dictionary
        kpPBZ_unfolded = {k: v for val in self.kpSBZcalc.values() for k, v in val.items()} 
        
        # Sort the unfolded PBZ
        kpPBZ_unfolded = {k: v for k, v in sorted(kpPBZ_unfolded.items(), key=lambda item: item[0])}

        # This makes sure when you do section by section DFT band structures
        # then read only read part of it.
        # k-index, k1, k2, k3
        self.unfolded_kpts_dat = np.array([[kk]+list(self.kpointsPBZ_full[kk, :3]) 
                                           for kk in kpPBZ_unfolded])
        
        # Generate k-path in distance unit
        self.kpline = bandstructure.KPOINTSline(kpred=self.unfolded_kpts_dat[:,1:], 
                                                supercell=self.transformation_matrix,
                                                breakTHRESH=self.kline_discontinuity_threshold)
        
        # Insert k on path (A^-1) after k-indices
        self.unfolded_kpts_dat = np.insert(self.unfolded_kpts_dat,1, self.kpline, axis=1)
 
        # Add k-indices and k-path to the energy, weight band structure array
        self.unfolded_bandstructure = np.concatenate([np.insert(unf, [0, 0], [kk, self.kpline[ii]], axis=1) 
                                                      for ii, (kk, unf) in enumerate(kpPBZ_unfolded.items())], axis=0)
        # Print information about folding
        if self.print_information is not None: 
            self._print_info_post(level=self.print_information)          
        return
    
    def _print_Post_unfolded_PBZ_kpts(self):
        """
        Print the unfolded k-points.

        Returns
        -------
        None.

        """
        print(f"{'='*_draw_line_length}\n- Unfolded PC k-points from postprocessed wavefunction file:[k-index, k on path (A^-1), k1, k2, k3]")
        for val in self.unfolded_kpts_dat:
                print('-- ' + f"{int(val[0]):5d}" + "  ".join(f"{x:12.8f}" for x in val[1:]))

    def _print_info_post(self, level='low'):
        """
        Printing information about the unfolding (e.g. unfolded k-points).

        Parameters
        ----------
        level : ['low','medium','high'], optional
            Level of printing information. The default is 'low'.

        Returns
        -------
        None.

        """
        if level == 'high':
            self._print_Post_unfolded_PBZ_kpts()
            
    def _save_unfolded_kpts_bandstr(self, save_unfolded_kpts, save_unfolded_bandstr): 
        """
        Save the unfolded kpoints and band structure data.

        Parameters
        ----------
        save_unfolded_kpts : dictionary, optional
            save2file :: Save unfolded kpoints data to file or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name.
            The default is {'save2file': False, 'fdir': '.', 'fname': 'kpoints_unfolded', 'fname_suffix': ''}.
        save_unfolded_bandstr : dictionary, optional
            save2file :: Save unfolded bandstructure data to file or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name. 
            The default is {'save2file': False, 'fdir': '.', 'fname': 'bandstructure_unfolded', 'fname_suffix': ''}.

        Returns
        -------
        None.

        """
        # Save unfolded kpoints after post processing of wave function file
        save_unfolded_kpts = _SaveData2File._default_unfolded_kp_bd_save_settings(save_unfolded_kpts)
        if save_unfolded_kpts['save2file']:
            self._save_Post_unfolded_PBZ_kpts(unfolded_kpts_dat=self.unfolded_kpts_dat, 
                                              save_dir=save_unfolded_kpts["fdir"], 
                                              file_name=save_unfolded_kpts["fname"], 
                                              file_name_suffix=save_unfolded_kpts["fname_suffix"],
                                              print_information=self.print_information)
        # Save unfolded band structure after post processing of wave function file
        save_unfolded_bandstr = _SaveData2File._default_unfolded_kp_bd_save_settings(save_unfolded_bandstr)
        if save_unfolded_bandstr['save2file']:
            self._save_Post_unfolded_bandstucture(unfolded_bandstructure=self.unfolded_bandstructure, 
                                                  save_dir=save_unfolded_bandstr["fdir"], 
                                                  file_name=save_unfolded_bandstr["fname"], 
                                                  file_name_suffix=save_unfolded_bandstr["fname_suffix"],
                                                  print_information=self.print_information)

    def _unfold(self, bandstructure): 
        """

        Parameters
        ----------
        bandstructure : BandStructure instance from irrep.bandstructure 
            BandStructure instance from irrep.bandstructure class.
            
        Returns
        -------
        numpy ndarray
            Unfolded effective band structure.
            Format: k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if is_spinor.
        numpy ndarray
            Unfolded effective band structure k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]

        """
        # Unfold bandstructure
        self._unfold_bandstructure(bandstructure)
        
        # Save unfolded kpoints after post processing of wave function file
        self._save_unfolded_kpts_bandstr(self.save_unfolded_kpts, self.save_unfolded_bandstr)
        
        return self.unfolded_bandstructure, self.unfolded_kpts_dat
