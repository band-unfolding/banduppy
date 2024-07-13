import numpy as np
from ..BasicFunctions.general_functions import _SaveData2File, _BasicFunctionsModule, _draw_line_length
try:
    from irrep.__aux import is_round
except ImportError:
    from irrep.utility import is_round

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
        header_msg  = " Unfolded PC k-points from postprocessed wavefunction file\n"
        header_msg += " k-index, k on path (A^-1), k1, k2, k3\n"
        # Save the sc-kpoints in file
        save_f_name = \
        _SaveData2File._save_2_file(data=unfolded_kpts_dat, 
                                    save_dir=save_dir, file_name=file_name,
                                    file_name_suffix=f'{file_name_suffix}.dat', 
                                    header_txt=header_msg, comments_symbol='#',
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
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor]
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
        header_msg  = " Unfolded band structure from postprocessed wavefunction file\n"
        header_msg += " k-index, k on path (A^-1), energy, weight " + \
                        ("Sx,Sy,Sz" if is_spinor else "")  +"\n"
        # Save the unfolded band structure in file
        save_f_name = \
        _SaveData2File._save_2_file(data=unfolded_bandstructure, 
                                    save_dir=save_dir, file_name=file_name,
                                    file_name_suffix=f'{file_name_suffix}.dat', 
                                    header_txt=header_msg, comments_symbol='#',
                                    print_log=bool(print_information)) #,
                                      #np_data_fmt='%.18e')
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
    
    def _gather_generated_ab_calculated_kpts(self, wavefns_file_kpts):
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
                if is_round(KP.K - self.kpointsSBZ[key, :3], prec = 1e-6):
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
                
    def _generate_kpoints_line(self, irrep_bandstr_instance, pc_kpts_coord, 
                               transformation_matrix, breakTHRESH = 0.1):
        """
        Generates kpoints line (after cartesian coordinate conversion).
        k_vec = u.b1 + v.b2 + w.b3
        e.g. (u,v,w)=(1/2, 0, 0) -> k_vec==(kx, ky, kz)
        [kx ky kz] = [u v w].[[b11 b12 b13], [b21 b22 b23], [b31 b32 b33]]

        Parameters
        ----------
        irrep_bandstr_instance : irrep.bandstructure.BandStructure
            irrep.bandstructure.BandStructure instance.
        pc_kpts_coord : ndarray of floats
            PC kpoints list.
        transformation_matrix : 3X3 matrix
            Primitive-to-supercell transformation matrix.
        breakTHRESH : float, optional
            If the distance between two neighboring k-points in the path is 
            larger than `break_thresh` break continuity in k-path. Set break_thresh 
            to a large value if the unfolded kpoints line is continuous.
            The default is 0.1.

        Returns
        -------
        K : numpy 1d array of floats
            Primitive cell k-points line path for unfolded bandstructure.

        """
        # Supercell lattice vectors in real space
        real_lattice_sc = irrep_bandstr_instance.Lattice
        # Reciprocal lattice super cell
        reciprocal_lattice_sc = 2*np.pi*np.linalg.pinv(real_lattice_sc).T
        # Reciprocal lattice primitive cell
        reciprocal_lattice_pc = transformation_matrix.T @ reciprocal_lattice_sc
        # Convert k-points to inverse angstrom unit
        KPcart = np.dot(pc_kpts_coord, reciprocal_lattice_pc)
        #KPcart = np.linalg.solve(np.dot(sc_lattice, np.linalg.pinv(transformation_matrix)), pc_kpts_coord.T).T # without 2*pi
        
        # Generate distance array
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > breakTHRESH] = 0.0
        K[1:] = np.cumsum(k)
        return K
        
    def _unfold_bandstructure(self, bandstructure, break_thresh = 0.1):  
        """
        

        Parameters
        ----------
        bandstructure : irrep.bandstructure.BandStructure
            irrep.bandstructure.BandStructure instance.
        break_thresh : float, optional
            If the distance between two neighboring k-points in the path is 
            larger than `break_thresh` break continuity in k-path. Set break_thresh 
            to a large value if the unfolded kpoints line is continuous.
            The default is 0.1.

        Returns
        -------
        None.

        """            
        # Check/return if kpoints in the generated SC-kpoints list exists in the
        # wavefunction file from ab-initio calculation.
        _ = self._gather_generated_ab_calculated_kpts(bandstructure.kpoints)
        
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
        self.kpline = self._generate_kpoints_line(bandstructure, self.unfolded_kpts_dat[:,1:], 
                                                  self.transformation_matrix, breakTHRESH=break_thresh)
        # Insert k on path (A^-1) after k-indices
        self.unfolded_kpts_dat = np.insert(self.unfolded_kpts_dat,1, self.kpline, axis=1)
        
        # Add k-indices and k-path to the energy, weight band structure array
        self.unfolded_bandstructure = np.concatenate([np.insert(unf, [0, 0], [kk, self.kpline[kk]], axis=1) 
                                                      for kk, unf in kpPBZ_unfolded.items()], axis=0)
        
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
        print(f"{'='*_draw_line_length}\n- Unfolded PC k-points from postprocessed wavefunction file:[k on path (A^-1), k1, k2, k3]")
        for val in self.unfolded_kpts_dat:
                print('-- '+"  ".join(f"{x:12.8f}" for x in val[1:]))

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
            
    def _save_unfolded_kpts_bandstr(self,
                                    save_unfolded_kpts = {'save2file': False,
                                                          'fdir': '.',
                                                          'fname': 'kpoints_unfolded',
                                                          'fname_suffix': ''},
                                    save_unfolded_bandstr = {'save2file': False, 
                                                             'fdir': '.',
                                                             'fname': 'bandstructure_unfolded',
                                                             'fname_suffix': ''}): 
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
        save_unfolded_kpts = _SaveData2File._default_save_settings(save_unfolded_kpts)
        if save_unfolded_kpts['save2file']:
            self._save_Post_unfolded_PBZ_kpts(unfolded_kpts_dat=self.unfolded_kpts_dat, 
                                              save_dir=save_unfolded_kpts["fdir"], 
                                              file_name=save_unfolded_kpts["fname"], 
                                              file_name_suffix=save_unfolded_kpts["fname_suffix"],
                                              print_information=self.print_information)
        # Save unfolded band structure after post processing of wave function file
        save_unfolded_bandstr = _SaveData2File._default_save_settings(save_unfolded_bandstr)
        if save_unfolded_bandstr['save2file']:
            self._save_Post_unfolded_bandstucture(unfolded_bandstructure=self.unfolded_bandstructure, 
                                                  save_dir=save_unfolded_bandstr["fdir"], 
                                                  file_name=save_unfolded_bandstr["fname"], 
                                                  file_name_suffix=save_unfolded_bandstr["fname_suffix"],
                                                  print_information=self.print_information)

    def _unfold(self, bandstructure, kline_discontinuity_threshold = 0.1,
                save_unfolded_kpts = {'save2file': False, 
                                      'fdir': '.',
                                      'fname': 'kpoints_unfolded',
                                      'fname_suffix': ''},
                save_unfolded_bandstr = {'save2file': False, 
                                         'fdir': '.',
                                         'fname': 'bandstructure_unfolded',
                                         'fname_suffix': ''}): 
        """
        

        Parameters
        ----------
        bandstructure : irrep.bandstructure.BandStructure
            irrep.bandstructure.BandStructure.
        kline_discontinuity_threshold : float, optional
            If the distance between two neighboring k-points in the path is 
            larger than `break_thresh` break continuity in k-path. Set break_thresh 
            to a large value if the unfolded kpoints line is continuous.
            The default is 0.1.
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
            save2file :: Save unfolded band structure data to file or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name. 
            The default is {'save2file': False, 'fdir': '.', 'fname': 'bandstructure_unfolded', 'fname_suffix': ''}.
            
        Returns
        -------
        numpy ndarray
            Unfolded effective band structure.
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
        numpy ndarray
            Unfolded effective band structure k-path.
            Format: k on path (A^-1)

        """
        # kline_discontinuity_threshold: float, default=0.1
        #    If the distance between two neighboring k-points in the path is 
        #    larger than `kline_discontinuity_threshold`, it is taken to be 0.
        # Set kline_discontinuity_threshold to a large value if the unfolded kpoints line is continuous
        # Unfold bandstructure
        self._unfold_bandstructure(bandstructure, break_thresh=kline_discontinuity_threshold)
        
        # Save unfolded kpoints after post processing of wave function file
        self._save_unfolded_kpts_bandstr(save_unfolded_kpts, save_unfolded_bandstr)
        
        return self.unfolded_bandstructure, self.kpline
