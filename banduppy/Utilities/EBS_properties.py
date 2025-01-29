import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import distance
from ..BasicFunctions.general_functions import _SaveData2File, _draw_line_length

### ===========================================================================
class _GeneralFunctionsDefs:
    @staticmethod
    def _reformat_columns_full_bandstr_data(full_data, is_band_center_data:bool=False):
        """
        Gather first 4 (5 for band centers) columns of data. Discard S_x,y,z data in spinor.

        Parameters
        ----------
        full_data : ndarray
            Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor.] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        is_band_center_data : bool, optional
            Is the data for unfolded band center? The default is False.
        
        Returns
        -------
        ndarray
            Unfolded effective band structure/band center data within the range. 
            Format: [k index, k on path (A^-1), energy (eV), weight] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers

        """
        if is_band_center_data:
            return full_data[:, :5] # column upto Sum of dN
        return full_data[:, :4] # column upto weight
        
    @classmethod
    def _get_data_in_energy_window(cls, full_data, Ef, Emin=None, Emax=None,  
                                   pad_energy_scale:float=0.5, 
                                   threshold_weight:float=None):
        """
        Collect data within the condition and range specified.

        Parameters
        ----------
        full_data : ndarray, optional
            Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        Ef : float
            Fermi energy. Set to 0.0 if None or 'auto'.
        Emin : float, optional
            Minimum in energy. The default is None.
        Emax : float, optional
            Maximum in energy. The default is None.
        pad_energy_scale: float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        threshold_weight : float, optional
            The band centers with band weights lower than the threshhold weights 
            are put to zero. The default is None. If None, this is ignored.
            
         Returns
         -------
         Emin : float
             Minimum in energy. 
         Emax : float
             Maximum in energy.
         result: ndarray
             Unfolded effective band structure/band center data within the range. 

         """
        if Ef == 'auto' or Ef is None:  Ef = 0.0
        # Shift the energy scale to 0 fermi energy level   
        if Ef is not None:
            YYY = full_data[:, 2] - Ef 
            print(f"-- Efermi was set to {Ef:.4f} eV")

        if Emin is None: Emin = YYY.min() - pad_energy_scale
        if Emax is None: Emax = YYY.max() + pad_energy_scale
        
        pos_right_energy_window = (YYY >= Emin) * (YYY <= Emax)
        result = full_data[pos_right_energy_window]
        result[:, 2] = YYY[pos_right_energy_window]

        # Set weights to 0 which are below threshold_weight
        if threshold_weight is not None: 
            result[result[:, -1] < threshold_weight, -1] = 0
            
        return Emin, Emax, result
    
    @classmethod 
    def _get_bandstr_data_only_in_energy_kpts_window(cls, complete_data, Ef:float=None, 
                                                     Emin:float=None, Emax:float=None,  
                                                     pad_energy_scale:float=0.5, 
                                                     is_band_center_data:bool=False,
                                                     min_dN_screen:float=0.0):
        """
        Collect data within the condition and range specified. 
        Note: Returns only 1st 4 (5 for band centers) columns. Removes the spinor data part for 
        spinor activated effective band structure data.

        Parameters
        ----------
        complete_data : ndarray, optional
            Unfolded effective band structure/band center data. 
        Ef : float
            Fermi energy. Set to 0.0 if None or 'auto'.
        Emin : float, optional
            Minimum in energy. The default is None.
        Emax : float, optional
            Maximum in energy. The default is None.
        pad_energy_scale: float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        is_band_center_data : bool, optional
            Is the data for unfolded band center? The default is False.
        min_dN_screen : float, optional
            The band centers with band weights lower than the threshhold weights 
            are discarded. The default is 0.
            
         Returns
         -------
         Emin : float
             Minimum in energy.
         Emax : float
             Maximum in energy
         ndarray
             Unfolded effective band structure/band center data within the range. 

         """
        
        # Returns only 1st 4 columns. Removes the spinor data part.
        complete_data_ = cls._reformat_columns_full_bandstr_data(complete_data, 
                                                                 is_band_center_data=is_band_center_data)
        # Collect data within energy window
        Emin, Emax, complete_data_ = \
            cls._get_data_in_energy_window(complete_data_, Ef, Emin=Emin, Emax=Emax,  
                                           pad_energy_scale=pad_energy_scale)
        # Pre-screen band structure data to minimize unnecessary small weights bands
        return Emin, Emax, complete_data_[complete_data_[:, -1] >= min_dN_screen]
    
    @classmethod
    def _save_band_centers(cls, data2save, print_log='low', 
                           save_data_f_prop = {'save2file': False, 'fdir': '.',
                                               'fname': 'unfolded_bandcenters',
                                               'fname_suffix': ''}):
        """
        Save unfolded band centers data.
        Format: [k index, k on path (A^-1), Band center (eV), Band width (eV), Sum of dN]

        Parameters
        ----------
        data2save : 2d array
            Data to save.
        print_log : [None,'low','medium','high'], optional
            Level of printing information. 
            The default is 'low'. If None, nothing is printed.
        save_data : dictionary, optional
            save2file :: Save data to file or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name.
            The default is {'save2file': False, 'fdir': '.', 'fname': 'unfolded_bandcenters', 'fname_suffix': ''}.

        Returns
        -------
        None.

        """  
        save_data = _SaveData2File._default_save_settings(save_data_f_prop)
        if save_data['save2file']:     
            if print_log is not None: 
                print(f"{'='*_draw_line_length}\n- Saving unfolded band centers to file...")
            header_msg  = " Unfolded band centers data\n"
            header_msg += "k-index, k on path (A^-1), Band center energy (eV), Band width (eV), Sum of dN\n"
            # Save the sc-kpoints in file
            save_f_name = _SaveData2File._save_2_file(data=data2save, 
                                                      save_dir=save_data["fdir"], 
                                                      file_name=save_data["fname"],
                                                      file_name_suffix=f'{save_data["fname_suffix"]}.dat', 
                                                      header_txt=header_msg, comments_symbol='#',
                                                      print_log=bool(print_log))
            if print_log is not None: 
                print(f'-- Filepath: {save_f_name}\n- Done')
        return

class _FormatSpecialKpts:
    """
    Find position and format labels of the special k-points.

    """
    @staticmethod
    def _extract_special_kpts_info(special_kpts, kpath_angs):
        """
        Reform position and labels of special kpoints.
    
        Parameters
        ----------
        special_kpts : dictionary
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. 
        kpath_angs : array
            k on path (in A^-1) coordinate.
    
        Returns
        -------
        special_kpts_labels : string list
            Labels of the special kpoints.
        special_kpts_poss : float list
            Positions (in angstrom) of the special kpoints.
    
        """
        kl = np.array([kpath_angs[ik] for ik in special_kpts.keys()])
        ll = np.array([k for k in special_kpts.values()])
        borders = [0] + list(np.where((kl[1:]-kl[:-1])>1e-4)[0]+1) + [len(kl)]
        k_labels=[(kl[b1:b2].mean(),"/".join(list(dict.fromkeys(ll[b1:b2])))) for b1,b2 in zip(borders,borders[1:])]
        
        special_kpts_labels = [label[1] for label in k_labels]
        special_kpts_poss = [label[0] for label in k_labels]
        return special_kpts_labels, special_kpts_poss

class _BandCentersBroadening(_GeneralFunctionsDefs):
    """
    Find band centers and broadening of the unfolded band structure. 
    
    The implementation is based on the SCF algorithm of automatic band center 
    determination from the following references:
    1. Medeiros et al, PRB 89, 041407(R) (2014) 
    2. Mondal et al, TBA
    
    Original implementation: 
        https://github.com/band-unfolding/bandup/utils/post_unfolding/
        locate_band_centers_and_estimate_broadening/find_band_centers_and_broadenings.py

    """
    def __init__(self, unfolded_bandstructure, 
                 algorithm_version:str='Mondal2025',
                 sigma:float=1,
                 min_sum_dNs_for_a_band:float=0.1,
                 min_dN_pre_screening:float=1e-2,
                 threshold_dN_2b_trial_band_center:float=None,
                 precision_scf_band_centers:float=1e-5,
                 err_tolerance_compare_kpts_val:float=1e-8,
                 print_log='low'):
        """
        Intializing BandCentersBroadening class. Play with the different cut-offs
        and threshold values here.

        Parameters
        ----------
        unfolded_bandstructure : numpy array
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if is_spinor]
        algorithm_version : str, optional
            Algorithm to determine band centers and broadening. 
            Options are: 'Medeiros2014','Mondal2025'
            The default is 'Mondal2025'.
            Note: 
            1. Medeiros2014 algorithm uses dN as weights during band center determination.
            o average_bandcenter = average(energies, weights=dN)
            2. Mondal2025 version additionally uses distance weights (Gaussian decay 
            around average reference band center).
            o distace_2_ref = average(energies, weights=dN)
            o weights_distance = exp(-(distace_2_ref**2)/(2*sigma**2))
            o average_bandcenter = average(energies, weights=dN*weights_distance) 
        sigma : float, optional
            Standard deviation of Gaussian decay for the weights during average band center
            determination. The decay is based on energy axis. The default is 1.
            Note: This parameter is used in 'Mondal2025' algorithm_version. 
            o distace_2_ref = average(energies, weights=dN)
            o weights_distance = exp(-(distace_2_ref**2)/(2*sigma**2))
            o average_bandcenter = average(energies, weights=dN*weights_distance)
        min_sum_dNs_for_a_band : float, optional
            Cut off criteria for minimum weights that a band center should have. 
            The band centers with lower weights than min_sum_dNs_for_a_band will be
            discarded during SCF refinements. If min_sum_dNs_for_a_band  
            is smaller than threshold_dN_2b_trial_band_center, min_sum_dNs_for_a_band
            will be reset to threshold_dN_2b_trial_band_center value.
            The default is 1e-1. 
        min_dN_pre_screening : float, optional
            Discard the bands which has weights below min_dN_pre_screening to start with. 
            This pre-screening step helps to minimize the data that will processed
            now on. The default is 1e-2. 
        threshold_dN_2b_trial_band_center : float, optional
            Initial guess of the band centers based on the threshold wights. If None,
            all the band centers are used as the initial guess. The default is None. 
        precision_scf_band_centers : float, optional
            Precision when compared band centers from previous and current SCF
            iteration. SCF is considered converged if this precision is reached.
            The default is 1e-5.
        err_tolerance_compare_kpts_val : float, optional
            The tolerance to group the bands set per unique kpoints. This
            determines if two flotting point numbers are the same or not. This is not 
            a critical parameter for band center determination algorithm.
            The default is 1e-8. 
        print_log : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """
        # Returns only 1st 4 columns. Removes the spinor data part.
        # Pre-screen band structure data to minimize unnecessary small weights bands
        self.unfolded_bandstructure_ = \
            self._pre_screen_bandstr_data(_GeneralFunctionsDefs._reformat_columns_full_bandstr_data(unfolded_bandstructure, 
                                                                                                    is_band_center_data=True), 
                                          min_dN_pre_screening)
        
        # Setting parameters
        self.use_algorithm = algorithm_version
        self.sigma = sigma
        self.prec_scf_band_centers = precision_scf_band_centers
        self.err_tolerance = err_tolerance_compare_kpts_val
        self.print_output = print_log
        
        # Reset some parameters based on condition check
        self.min_sum_dNs_for_each_band, self.threshold_dN_trial_band_center = \
            self._check_min_sum_dNs_condition(min_sum_dNs_for_a_band, 
                                              threshold_dN_2b_trial_band_center)
        
    def _scfs_band_centers_broadening(self, collect_data_scf:bool=False,
                                      save_data = {'save2file': False, 
                                                   'fdir': '.',
                                                   'fname': 'unfolded_bandcenters',
                                                   'fname_suffix': ''}):
        """
        Find the band centers and broadening for a band structure. 
        
        The implementation is based on the SCF algorithm of automatic band center 
        determination from the following references:
        1. Medeiros et al, PRB 89, 041407(R) (2014) 
        2. Mondal et al,
        
        Original implementation: 
            https://github.com/band-unfolding/bandup/utils/post_unfolding/
            locate_band_centers_and_estimate_broadening/find_band_centers_and_broadenings.py

        Parameters
        ----------
        collect_data_scf : bool, optional
            Whether to save the dtails of band centers in each SCF cycles. 
            The default is False.
        save_data : dictionary, optional
            save2file :: Save unfolded band centers data to file or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name.
            The default is {'save2file': False, 'fdir': '.', 'fname': 'unfolded_bandcenters', 'fname_suffix': ''}.

        Returns
        -------
        return_grouped_unfolded_bandstructure_datails : 2d array
            Each array contains the final details of band centers at each
            kpoint. 
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN]
        gather_scf_data_ : dictionary of dictionary of array or None
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_data_scf is false.
            Format: {kpoint_index: {SCF_cycle_index: [Band center, Band width, Sum of dN]}}

        """
        print(f"{'-'*72}\n- Finding band center and broadening for band structure...")
        # self.unfolded_bandstructure_ = self._eliminate_duplicate_kpts_list(self.unfolded_bandstructure_)
        grouped_unfolded_bandstructure = self._group_unique_kpts_data(self.unfolded_bandstructure_,
                                                                      err_tolerance=self.err_tolerance)
        gather_scf_data_ = {} if collect_data_scf else None
        return_grouped_unfolded_bandstructure_datails = []
        
        for unfolded_bandstructure_current_kpt in  grouped_unfolded_bandstructure:
            k_index_i, guess_band_details, all_data_ = \
                self._scfs_band_centers_broadening_kpt(unfolded_bandstructure_current_kpt,
                                                       collect_data_scf=collect_data_scf)
            if collect_data_scf: gather_scf_data_[k_index_i] = all_data_
            return_grouped_unfolded_bandstructure_datails.append(guess_band_details)
        print('- Done')   
        #return return_grouped_unfolded_bandstructure_datails, gather_scf_data_
        final_results = np.concatenate(return_grouped_unfolded_bandstructure_datails, axis=0)
        # Save the band ceneters data
        _GeneralFunctionsDefs._save_band_centers(data2save=final_results, 
                                                 print_log=self.print_output,
                                                 save_data_f_prop=save_data)    
        
        return final_results, gather_scf_data_
        
    def _scfs_band_centers_broadening_kpt(self, unfolded_bandstructure_current_kpt,
                                          collect_data_scf:bool=False):
        """
        Find the band centers and broadening for a particular kpoint.

        Parameters
        ----------
        unfolded_bandstructure_current_kpt : 2d numpy array
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy (eV), weight]
        collect_data_scf : bool, optional
            Whether to save the dtails of band centers in each SCF cycles. 
            The default is False.

        Returns
        -------
        k_index_ : int
            Index of the kpoint.
        guess_band_details : 2d numpy array
            Band center details at the particular kpoint.
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN]
        all_data_ : dict or None
            Dictionary containing the band center details of all SCF cycles. 
            Return None if collect_data_scf is False.
            Format: {SCF_cycle_index: [Band center, Band width, Sum of dN]}

        """
        # Collect kpoint information
        kpoints_cord = unfolded_bandstructure_current_kpt[0, :2]
        k_index_ = int(kpoints_cord[0])
        if self.print_output in ['low','medium','high']:
            print (f"{'-'*72}\n- Finding band center for kpoint: {k_index_}")
        all_data_ = {} if collect_data_scf else None
        
        # Collect dNs and energies in array
        self.dNs_for_current_kpt = unfolded_bandstructure_current_kpt[:, -1].copy()
        self.energies_for_current_kpt = unfolded_bandstructure_current_kpt[:, 2].copy()
        self._min_energy, self._max_energy = min(self.energies_for_current_kpt), max(self.energies_for_current_kpt)

        # Initialize gussed band centers
        ## Apply threshold dN for trial band center
        if self.threshold_dN_trial_band_center is None:
            guess_band_centers = self.energies_for_current_kpt.copy()
        else:
            guess_band_centers = self.energies_for_current_kpt[self.dNs_for_current_kpt >=
                                                               self.threshold_dN_trial_band_center]
        n_guesses_bc_start = len(guess_band_centers)

        # Run self-consistence loop
        self.scf_loop_count = 0
        converged = False
        while(not converged):
            self.scf_loop_count += 1
            if self.print_output == 'high': print(f'-- SCF cycle: {self.scf_loop_count}')
            guess_band_details = self._calculate_guess_band_details(guess_band_centers)
            if len(guess_band_details) < 1:
                print('Warning: No band center found at this k-point.')
                break
            refined_band_centers = self._refine_band_centers(guess_band_details)
            converged = self._check_convergence(guess_band_centers, refined_band_centers, 
                                                self.prec_scf_band_centers)
            if converged:
                guess_band_details = self._calculate_guess_band_details(refined_band_centers)
                guess_band_details = guess_band_details[guess_band_details[:, -1] >= 
                                                        self.min_sum_dNs_for_each_band]
                guess_band_details = np.insert(guess_band_details, [0, 0], kpoints_cord[:2], axis=1)
                n_guesses_bc_end = len(guess_band_details)
                if self.print_output == 'high':
                    print('-- Positions of the band centers converged:')
                    print(f'\t--- Precision reached: {1000.0 * self.prec_scf_band_centers} meV')
                    print(f'\t--- Total SCF steps: {self.scf_loop_count}')
                    print(f'\t--- Start number of band centers: {n_guesses_bc_start}')
                    print(f'\t--- Final number of band centers: {n_guesses_bc_end}')
            else:
                guess_band_centers = refined_band_centers
                
            if collect_data_scf: all_data_[self.scf_loop_count] = guess_band_details[:, -3:]
        # guess_band_details = (#kpoint index #kpoint coordinate #Band center #Band width #Sum of dN)
        return k_index_, guess_band_details, all_data_
    
    @staticmethod
    def _check_min_sum_dNs_condition(min_sum_dNs_for_a_band, threshold_dN_2b_trial_band_center):
        """
        Checking cut off criteria for minimum weights that band center should have. 
        The band centers with lower weights than min_sum_dNs_for_a_band will be
        discarded during SCF refinements. If min_sum_dNs_for_a_band  
        is smaller than threshold_dN_2b_trial_band_center, min_sum_dNs_for_a_band
        will be reset to threshold_dN_2b_trial_band_center value.

        Parameters
        ----------
        min_sum_dNs_for_a_band : float
            Cut off criteria for minimum weights that band center should have. 
            The band centers with lower weights than min_sum_dNs_for_a_band will be
            discarded during SCF refinements.
        threshold_dN_2b_trial_band_center : float
            Initial guess of the band centers based on the threshold wights. If None,
            do nothing.

        Returns
        -------
        min_sum_dNs_for_a_band : float
            Cut off criteria for minimum weights that band center should have. 
            The band centers with lower weights than min_sum_dNs_for_a_band will be
            discarded during SCF refinements.
        threshold_dN_2b_trial_band_center : float
            Initial guess of the band centers based on the threshold wights.

        """
        if threshold_dN_2b_trial_band_center is not None:
            threshold_dN_2b_trial_band_center = abs(threshold_dN_2b_trial_band_center)
            if(abs(min_sum_dNs_for_a_band) < threshold_dN_2b_trial_band_center):
                min_sum_dNs_for_a_band = threshold_dN_2b_trial_band_center
                print('- WARNING: Resetting min_sum_dNs_for_a_band because it is smaller than threshold_dN_2b_trial_band_center.')
        return min_sum_dNs_for_a_band, threshold_dN_2b_trial_band_center
    
    @classmethod
    def _pre_screen_bandstr_data(cls, unfolded_bandstructure, min_dN):
        """
        Pre-screen band structure data. Discard unnecessary band centers which
        has very small weights. This minimizes the search space and improves efficiency.

        Parameters
        ----------
        unfolded_bandstructure : 2d numpy array
            Unfolded effective band structure before removing small weights centers. 
            Format: [k index, k on path (A^-1), energy (eV), weight]
        min_dN : float
            Discard the bands which has weights below min_dN_pre_screening. This
            pre-screening step helps to minimize the data that will processed
            now on.

        Returns
        -------
        2d numpy array
            Unfolded effective band structure after removing small weights centers. 
            Format: [k index, k on path (A^-1), energy (eV), weight]

        """
        # Pre-screening: get rid of very small weights
        return unfolded_bandstructure[unfolded_bandstructure[:, -1] >= min_dN]

    @classmethod
    def _eliminate_duplicate_kpts_list(cls, unfolded_bandstructure):
        """
        Eliminating duplicated points from the unfolded band structure data. In
        the way the band structure data are sorted. 
        Sort order: first kpts, then energy 
        e.g.: L-G,G-X -> get rid of two Gs

        Parameters
        ----------
        unfolded_bandstructure : 2d numpy array
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy, weight]

        Returns
        -------
        unfolded_bandstructure : 2d numpy array
            Unfolded effective band structure after removing duplicates. 
            Format: [k index, k on path (A^-1), energy, weight]

        """
        # Eliminating duplicated points, e.g.: L-G,G-X -> get rid of two Gs
        # Sort order: kpts, then energy
        sorted_index_data = np.lexsort((unfolded_bandstructure[:, 2], unfolded_bandstructure[:, 1]))
        unfolded_bandstructure = unfolded_bandstructure[sorted_index_data]
        dNs = unfolded_bandstructure[:, -1]
        for kp_point in range(len(unfolded_bandstructure)-1):
            kp_en1 = unfolded_bandstructure[kp_point]
            kp_en2 = unfolded_bandstructure[kp_point + 1]
            if(kp_en1[1]==kp_en2[1] and kp_en1[2]==kp_en2[2]):
                dNs[kp_point + 1] = max(kp_en1[3], kp_en2[3]) 
                dNs[kp_point] = 0.0
               
        unfolded_bandstructure[:, -1] = dNs
        return unfolded_bandstructure
    
    @classmethod
    def _group_unique_kpts_data(cls, unfolded_bandstructure, err_tolerance:float=1e-8):
        """
        Group bands by k-point indices.

        Parameters
        ----------
        unfolded_bandstructure : 2d numpy array
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy (eV), weight]
        err_tolerance : float, optional
            The tolerance to group the bands set per unique kpoints. 
            The default is 1e-8.

        Returns
        -------
        list
            List of unfolded effective band structure group by kpoints.
            Format: [k index, k on path (A^-1), energy (eV), weight]

        """
        # Get unique kpoints coordinate
        unique_kpts_coords = np.unique(unfolded_bandstructure[:, 0])
        # Group the rows based on unique values in the specified column
        return [unfolded_bandstructure[abs(unfolded_bandstructure[:, 0] - val) < err_tolerance] 
                for val in unique_kpts_coords]

    def _weighted_avg_and_std(self, values_, weights_):
        """
        Calculate wighted average and variance of list of values. For 'Mondal2025'
        algorithm the weights combines dNs and inverse_distnce.

        Parameters
        ----------
        values_ : 1d array or list
            Values to average.
        weights_ : 1d array or list
            Values to use as weights in averaging.

        Returns
        -------
        tuple (float, float, float)
            (average value, std deviation, total weight).

        """
        values_ = np.array(values_)
        weights_t = np.array(weights_).copy()
        if self.use_algorithm == 'Mondal2025':
            # Additionally apply distance weighted mean
            
            # # Version-1 implementation without refining the reference band center
            # #******************************************************************
            # ref_center_ = np.average(values_, weights=weights_t)
            # energy_distance = distance.cdist(values_[:, np.newaxis], [[ref_center_]]).flatten()
            # #==================================================================
            # # ------------ inverse distance decay ---------
            # # Note: Not quite worked out
            # #inverse_distance = 1/energy_distance
            # #distance_weights = inverse_distance / np.sum(inverse_distance[np.isfinite(inverse_distance)]) #normalize
            # # --------- exponential decay ------------
            # # Note: Too sensitive to the choice of decay coefficient
            # #distance_weights = np.exp((-1)*self.sigma*energy_distance)
            # # ------------- Gaussian decay -----------
            # distance_weights = np.exp(-(energy_distance*energy_distance)/(2*self.sigma*self.sigma))
            # #==================================================================
            # weights_t *= distance_weights
            # #******************************************************************
            
            # Version-2 implementation includs refining the reference band center
            ref_center_ = np.average(values_, weights=weights_t)
            ## refine the reference band center
            while True:
                ## Energy distance from reference point
                energy_distance = distance.cdist(values_[:, np.newaxis], [[ref_center_]]).flatten()
                ## Gaussian decay 
                distance_weights = np.exp(-(energy_distance*energy_distance)/(2*self.sigma*self.sigma))
                ## Refined band ceneter
                ref_center_new = np.average(values_, weights=weights_t*distance_weights)
                ## Check SCF
                if abs(ref_center_new - ref_center_) < self.prec_scf_band_centers:
                    weights_t *= distance_weights
                    break
                else:
                    ref_center_ = ref_center_new
           
        average = np.average(values_, weights=weights_t)
        variance = np.average((values_-average)**2, weights=weights_t) 
        total_weight = np.sum(weights_t)
        return (average, np.sqrt(variance), total_weight)  

    def _get_band_center_properties(self, bc_centers_, bc_weights_):
        """
        Calculates band ceneters properties.

        Parameters
        ----------
        bc_centers_ : 1d float array/list
            List of band ceneters to average.
        bc_weights_ : 1d array or list
            Values to use as weights in averaging.
            
        Returns
        -------
        list (float, float, float)
            (average band ceneter, band broadening/std deviation, total weight).

        """
        return self._weighted_avg_and_std(bc_centers_, bc_weights_)
    
    @classmethod
    def _calculate_possible_energy_width(cls, band_centers, min_energy, max_energy):
        """
        Set energy width at each band center.

        Parameters
        ----------
        band_centers : float array
            List of band ceneters.
        min_energy : float
            Minimum energy.
        max_energy : float
            Maximum energy.

        Returns
        -------
        2d numpy array
            Energy width at each band center [from, to].

        """
        XX = np.array(band_centers[:-1] + band_centers[1:]) * 0.5
        # +1.0 makes sure to cover '<guess_energy_width[iband][1]' condition
        # in calculate_guess_band_details()
        XX = np.insert(XX, [0, len(XX)], [min_energy, max_energy+1.0]) 
        return np.stack((XX[:-1], XX[1:]), axis=-1) 

    def _calculate_guess_band_details(self, guess_band_centers):
        """
        Calculates band ceneters, band weights, and band width.

        Parameters
        ----------
        guess_band_centers : float array
            List of guessed/refined band ceneters.
        self._min_energy : float
            Minimum energy.
        self._max_energy : float
            Maximum energy.
        self.energies_for_current_kpt : float array
            All bands energies at specific kpoint.
        self.dNs_for_current_kpt : float array
            All band weights at specific kpoint.

        Returns
        -------
        2d numpy array
            Band center details at the particular kpoint.
            Format: [Band center, Band width, Sum of dN]

        """
        guess_energy_width = self._calculate_possible_energy_width(guess_band_centers,
                                                                   self._min_energy, 
                                                                   self._max_energy)
        guess_band_details = [] 
        for iband in range(len(guess_band_centers)):
            indices_of_enegies_spread_in_band = \
                np.argwhere((self.energies_for_current_kpt >= guess_energy_width[iband][0]) & 
                            (self.energies_for_current_kpt < guess_energy_width[iband][1])).flatten()   
            if len(indices_of_enegies_spread_in_band) > 0:
                band_centers_, band_width_, band_weight_ = \
                    self._get_band_center_properties(self.energies_for_current_kpt[indices_of_enegies_spread_in_band], 
                                                     self.dNs_for_current_kpt[indices_of_enegies_spread_in_band])
                # Band center, standard width, weight
                guess_band_details.append([band_centers_, band_width_, band_weight_])
            
        return np.array(guess_band_details)

    def _refine_band_centers(self, guess_band_details):
        """
        Discard band centers which are too close in energy. Reducing the number 
        of too close energy values.
        
        The implementation is based on the SCF algorithm of automatic band center 
        determination from the following references:
        1. Medeiros et al, PRB 89, 041407(R) (2014) 
        2. Mondal et al,
        
        Original implementation: 
            https://github.com/band-unfolding/bandup/utils/post_unfolding/
            locate_band_centers_and_estimate_broadening/find_band_centers_and_broadenings.py
        
        Discard band centers with
        band_weight_current_band < self.min_sum_dNs_for_each_band     or
        abs(band_center_n,current - band_center_(n-l),current) < 2*max[band_n_width, band_n-l_width] 
        for l>=0 scuh that band_center_(n-l),current is an accepted band ceneter.

        Parameters
        ----------
        guess_band_details : 2d numpy array
            Band center details at the particular kpoint.
            Format: [Band center, Band width, Sum of dN]
        self.min_sum_dNs_for_each_band : float
            Cut off criteria for minimum weights that a band center should have. 
            The band centers with lower weights than self.min_sum_dNs_for_each_band will be
            discarded during SCF refinements.
        self.use_algorithm : str, optional
            Algorithm to determine band centers and broadening. 
            Options are: 'Medeiros2014','Mondal2025'

        Returns
        -------
        2d numpy array
            Band center details at the particular kpoint after refining.
            Format: [Band center, Band width, Sum of dN]

        """
        refined_band_centers = []
        iband_m_1 = -1
        low_wights_bands_merge_success = True
        for iband in range(len(guess_band_details)):
            # band center, band width, band Bloch weight
            bc_iband, bwidth_iband, bweight_iband = guess_band_details[iband] # current band
            if iband_m_1 is not None:
                bc_iband_m_1, bwidth_iband_m_1, bweight_iband_m_1 = guess_band_details[iband_m_1] #  iband_m_1 == current band minus 1
            if low_wights_bands_merge_success:
                tmp_bc_centers, tmp_bc_weights = [bc_iband_m_1], [bweight_iband_m_1]
            #print(iband, bc, bwidth, b_weight, bc_cbm1, bwidth_cbm1, b_weight_cbm1)
            valid_bc = False
            bc_diff_iband_ibandm1 = abs(bc_iband - bc_iband_m_1) #abs(band_center_n,current - band_center_(n-l),current)
            
            merge_band_centers_cond_bwidth = bc_diff_iband_ibandm1 < 2.0 * max([bwidth_iband, bwidth_iband_m_1])
            merge_band_centers_cond_bweights = bweight_iband < self.min_sum_dNs_for_each_band
            
            if merge_band_centers_cond_bwidth or merge_band_centers_cond_bweights: 
                if self.use_algorithm == 'Mondal2025':
                    # If we want to allow overlaps band widths not to merge
                    # if merge_band_centers_cond_bwidth and not merge_band_centers_cond_bweights: #(bweight_iband > 0.95):
                    #     valid_bc = True
                    # else:
                    tmp_bc_centers.append(bc_iband)
                    tmp_bc_weights.append(bweight_iband)
                    
                    bc_iband_m_1, bwidth_iband_m_1, bweight_iband_m_1 =\
                        self._get_band_center_properties(tmp_bc_centers,tmp_bc_weights)
                        
                    iband_m_1 = None
                    low_wights_bands_merge_success = False
    
                    if merge_band_centers_cond_bweights:
                        # This condition gives the lower weights a chance to merge and form a band center.
                        tmp_bc_iband_m_1, tmp_bwidth_iband_m_1, tmp_bweight_iband_m_1 =\
                            self._get_band_center_properties(tmp_bc_centers[1:],tmp_bc_weights[1:])
                        if (tmp_bweight_iband_m_1 > self.min_sum_dNs_for_each_band): 
                            bc_iband_m_1, bwidth_iband_m_1, bweight_iband_m_1 =\
                                tmp_bc_iband_m_1, tmp_bwidth_iband_m_1, tmp_bweight_iband_m_1
                            refined_band_centers.append(bc_iband_m_1)
                            low_wights_bands_merge_success = True
                    else:
                        if len(refined_band_centers) == 0:
                            refined_band_centers.append(bc_iband_m_1)
                        else:
                            refined_band_centers[-1] = bc_iband_m_1
                else:
                    try:
                        # 1. If n-1 band center has lower weight than current one then
                        # move the band center to the new one.
                        if(abs(bweight_iband / bweight_iband_m_1) > 1.0):
                            del refined_band_centers[-1]
                            valid_bc = True
                    except:
                        pass
            else:
                valid_bc = True

            if(valid_bc):
                low_wights_bands_merge_success = True
                iband_m_1 = iband
                refined_band_centers.append(bc_iband)
        return np.array(refined_band_centers)

    @staticmethod
    def _check_convergence(old_band_centers, new_band_centers, prec_scf_band_centers):
        """
        Check if the two band centers are close. If true, then the band ceneter
        is considered converged in SCF.

        Parameters
        ----------
        old_band_centers : 1d numpy array
            Previous band centers.
        new_band_centers : 1d numpy array
            Refined band centers.
        prec_scf_band_centers : float
            Precision when compared band centers from previous and latest SCF
            iteration. SCF is considered converged if this precision is reached.

        Returns
        -------
        bool
            If two arrays are same or not.

        """
        for ii, new_band_center_val in enumerate(new_band_centers):
            if abs(old_band_centers[ii] - new_band_center_val) > prec_scf_band_centers:
                return False
        return True

class _EffectiveMass:
    """
    Class for calculating effective mass.
    
    E (1 + gamma*E) = hbar^2 k^2/(2m*)
    => E (1 + gamma*E) = alpha k^2 
    => Solving for E yields a quadratic equation when gamma != 0.
    
    General equation: 
        E (1 + gamma*E) = alpha*(k-kshift)^2 + cbm
         
    Fit bandstructure along k-path to alpha and gamma. 
        alpha = hbar^2/(2m*) 
        gamma = non-parabolicity ==> gamma = 0 for parabolic bands
    Additional parameters:  
        kshift = horizontal kpath fit param 
        cbm = vertical energy fit param, conduction band minima
        
    o For E in eV and k in A^-1; alpha has unit of eV.A^2 and gamma has unit of eV^-1.
    
    STD in alpha --> STD in m*:
        std(m*) ~= |d_m*/d_alpha|.std(alpha)
                 = hbar^2/(2*m_e*alpha_in_eV_angsquare) * (1/alpha^2) * std(alpha)
                 = effective_mass_unit_conversion * (1/alpha^2) * std(alpha)
    
    """
    def __init__(self, print_log='low'):
        """
        Intializing EffectiveMass class. 
        
        m_e = 9.1093837015e-31 kg
        hbar = 1.054571817e-34 J.s
        1 ang = 1e-10 m
        1 eV = 1.602176634e-19 J
        
        effective_mass_unit_conversion = hbar^2/(2*m_e*alpha_in_eV_angsquare)
        = (1.054571817e-34**2)/(2*9.1093837015e-31*1.602176634e-19*1e-10**2) 
        = 3.8099821114859607
        
        Parameters
        ----------
        
        print_log : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.
        """
        self.print_output = print_log
        self.effective_mass_unit_conversion = 3.8099821114859607

    @classmethod
    def _fit_parabola_short(cls, k, alpha):
        return alpha * k * k

    @classmethod
    def _fit_hyperbola_positive_short(cls, k, alpha, gamma):
        # Positive solution of E quadratic equation
        return (-1 + np.sqrt(1 + 4*alpha*gamma*k*k))/(2*gamma)

    @classmethod
    def _fit_hyperbola_negative_short(cls, k, alpha, gamma):
        # Negative solution of E quadratic equation
        return (-1 - np.sqrt(1 + 4*alpha*gamma*k*k))/(2*gamma)

    @classmethod
    def _fit_parabola(cls, k, alpha, kshift, cbm):
        return alpha * np.power(k - kshift, 2) + cbm
    
    @classmethod
    def _fit_hyperbola_positive(cls, k, alpha, kshift, cbm, gamma):
        # Positive solution of E quadratic equation
        return (-1 + np.sqrt(1 + 4*alpha*gamma*np.power(k-kshift,2)))/(2*gamma) + cbm 
        
    @classmethod
    def _fit_hyperbola_negative(cls, k, alpha, kshift, cbm, gamma):
        # Negative solution of E quadratic equation
        return (-1 - np.sqrt(1 + 4*alpha*gamma*np.power(k-kshift,2)))/(2*gamma) + cbm 
        
    def _effective_mass_calculator(self, kpath, band_energy, ignore_kshift_cbm_fit:bool=True,
                                   p0=None, bounds = (-np.inf, np.inf), sigma=None,
                                   absolute_sigma:bool=False, fit_parabola:bool=False,
                                   fit_hyperbola_positive:bool=False, fit_hyperbola_negative:bool=False):
        """
        kpath : numpy array
            k on path (A^-1) of unfolded effective band structure/band centers that will
            be fitted for calculating effective mass.
        band_energy : numpy array
            Energy (in eV) of unfolded effective band structure/band centers that will
            be fitted for calculating effective mass.
        ignore_kshift_cbm_fit : bool, optional
            Ignore fitting kshift and cbm parameters during fitting. The default is True.
        p0 : array_like, optional
            Initial guess for the parameters (length N). If None, then the
            initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            ValueError is raised).
        bounds : 2-tuple of array_like or `Bounds`, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            There are two ways to specify the bounds:
                - Instance of `Bounds` class.
                - 2-tuple of array_like: Each element of the tuple must be either
                  an array with the length equal to the number of parameters, or a
                  scalar (in which case the bound is taken to be the same for all
                  parameters). Use ``np.inf`` with an appropriate sign to disable
                  bounds on all or some parameters.
        sigma : None or scalar or M-length sequence or MxM array, optional
            Determines the uncertainty in ydata. If we define residuals as 
            r = ydata - f(xdata, *popt), then the interpretation of sigma depends on 
            its number of dimensions:
            A scalar or 1-D sigma should contain values of standard deviations of 
            errors in ydata. In this case, the optimized function is 
            chisq = sum((r / sigma) ** 2). 
            A 2-D sigma should contain the covariance matrix of errors in ydata. 
            In this case, the optimized function is 
            chisq = r.T @ inv(sigma) @ r.
            The default is None. This is equivalent of 1-D sigma filled with ones.
        absolute_sigma : bool, optional
            If True, sigma is used in an absolute sense and the estimated parameter 
            covariance pcov reflects these absolute values.
            If False, only the relative magnitudes of the sigma values matter. 
            The default is False. 
        fit_parabola : bool, optional
            Fit parabolic Kane model of band dispersion. The default is False. 
            Order: fit_parabola > fit_hyperbola_positive > fit_hyperbola_negative
        fit_hyperbola_positive : bool, optional
            Fit hyperbolic model (upward hyperbola) of band dispersion. The default is False.
        fit_hyperbola_negative : bool, optional
            Fit hyperbolic model (downward hyperbola) of band dispersion. The default is False.

        Returns
        -------
        m_star : (float, float)
            Calculated effective mass (m_star[0]) and error (m_star[1]) in m_0 unit.
        popt : array
            Optimal values for the parameters so that the sum of the squared
            residuals of ``f(xdata, *popt) - ydata`` is minimized.
        pcov : 2-D array
            The estimated approximate covariance of popt. 
        params_errors : 1-D array
            One standard deviation error in parameters.
            perr = np.sqrt(np.diag(pcov))

        Raises
        ------
        ValueError
            if none of fit_* option is supplied.
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.
        RuntimeError
            if the least-squares minimization fails.
        OptimizeWarning
            if covariance of the parameters can not be estimated.
        """
        if fit_parabola:
            _fit_fn_call = self._fit_parabola_short if ignore_kshift_cbm_fit else self._fit_parabola
            popt, pcov = curve_fit(_fit_fn_call, kpath, band_energy, p0=p0,
                                   bounds=bounds, sigma=sigma, absolute_sigma=absolute_sigma)
        elif fit_hyperbola_positive:
            _fit_fn_call = self._fit_hyperbola_positive_short if ignore_kshift_cbm_fit else self._fit_hyperbola_positive
            popt, pcov = curve_fit(_fit_fn_call, kpath, band_energy, p0=p0,
                                   bounds=bounds, sigma=sigma, absolute_sigma=absolute_sigma)
        elif fit_hyperbola_negative:
            _fit_fn_call = self._fit_hyperbola_negative_short if ignore_kshift_cbm_fit else self._fit_hyperbola_negative
            popt, pcov = curve_fit(_fit_fn_call, kpath, band_energy, p0=p0,
                                   bounds=bounds, sigma=sigma, absolute_sigma=absolute_sigma)
        else:
            raise ValueError('No fitting option is supplied for fitting.')
            
        params_errors = np.sqrt(np.diag(pcov))
        m_star = self.effective_mass_unit_conversion/popt[0]
        m_star_error = self.effective_mass_unit_conversion*params_errors[0]/popt[0]/popt[0]
        return (m_star, m_star_error), popt, pcov, params_errors

class _alloy_scattering_params:
    """
    Class for calculating alloy-disordered scattering potential from band broadening.
    This calculates the alloy-disordered scattering lifetime
    using a fitted statistically averaged disorder potential in the matrix element for
    Fermi's golden rule. This analytical equation is fitted to the band broadening/width
    from first-principles band structure.
    
    o Analytical equation for scattering:
    1/tau = 2*pi/hbar * U0^2 *x(1-x) * Omega0 * m*^(3/2)/sqrt(2)/pi^2/hbar^3 * 
            (1+2*gamma*E) * sqrt(E(1+gamma*E)) * (1+2*gamma*E+(4/3)*gamma^2*E^2) / (1+2*gamma*E)^2
            
    o From first-principles band structure:
    Within the extended Ehrenreich and Schwartz theory, the alloy-disorder scattering
    rate is given by:
    1/tau = band_broadening / hbar
            
    The implementation is based on Reference: 
        1. Pant et. al., APL, 117, 242105 (2020)
        2. Mondal et al, TBA 
    """
    
    def __init__(self, m_star, unit_cell_volm, composition,
                 non_parabolocity_param=0):
        """
        

        Parameters
        ----------
        m_star : float 
            Carrier effective mass (in m0 unit). E.g. 0.2
        unit_cell_volm : float
            Primitive cell volume (in Angstrom^3).
        composition : float
            Alloy mole fraction (0 <= composition <= 1).
        non_parabolocity_param : float, optional
            hyperbolicity parameter / non-parabolic parameter, gamma, (eV^-1) 
            in Kane model for nonparabolic spherical bands:
            E (1 + gamma*E) = hbar^2 k^2/(2m*)
            The default is 0.

        Returns
        -------
        None.

        """
        self._m_star = m_star
        self._omega_0 = unit_cell_volm
        self._composition = composition
        self._alpha = non_parabolocity_param
        # sqrt(2)/pi/hbar^3*eV^2*Angstrom^3*m0^3/2*sqrt(eV)
        self._alloy_scattering_constant_fact = 0.021401081748922447

    def _Fermi_rule_fit(self, x, U0):
        """
        This function calculates the alloy-disordered scattering lifetime
        using a statistically averaged disorder potential in the matrix element for
        Fermi's golden rule.
        
        hbar*1/tau = 2*pi * U0^2 *x(1-x) * Omega0 * m*^(3/2)/sqrt(2)/pi^2/hbar^3 * 
                (1+2*gamma*E) * sqrt(E(1+gamma*E)) * (1+2*gamma*E+(4/3)*gamma^2*E^2) / (1+2*gamma*E)^2
                
        Reference: 
            1. Pant et. al., APL, 117, 242105 (2020)
            2. Mondal et al, TBA 

        Parameters
        ----------
        x : 1D array of floats 
            Band energies (eV) in the scattering lifetime equation.
        U0 : float
            Alloy-disordered scattering potential (eV).

        Returns
        -------
        one_by_tau : array of floats
            Alloy disordered scattering lifetime, hbar/tau (eV).

        """
        #_alloy_scattering_constant_fact = 0.021401081748922447 # sqrt(2)/pi/hbar^3*eV^2*Angstrom^3*m0^3/2*sqrt(eV)
        two_alpha_epsilon = 2*self._alpha*x
        I_prime = (1 + two_alpha_epsilon + (two_alpha_epsilon*two_alpha_epsilon/3))/(1+two_alpha_epsilon)
        one_by_tau = (self._alloy_scattering_constant_fact*U0*U0*
                      self._composition*(1-self._composition)*self._omega_0*
                      (self._m_star**(3/2))*np.sqrt(x*(1+self._alpha*x))*I_prime)
        return one_by_tau

    def _calculate_alloy_scattering_potential(self, band_energy, band_width, 
                                              intial_guess_u0=1, fitting_bounds_u0=(0,2)):
        """
        This function fits the analytical equation for scattering lifetime to the
        band broadening data obtained from first-principles calculations.

        Parameters
        ----------
        band_energy : float array
            Band energies (eV) to fit for scattering lifetime equation.
        band_width : float array
            Band width/broadening (eV) to fit for scattering lifetime equation..
        intial_guess_u0 : float, optional
            Intial guess for fitting parameter - scattering potential, U0. The default is 1.
        fitting_bounds_u0 : tuple, optional
            Fitting bounds for fitting parameter - scattering potential, U0. The default is (0,2).

        Returns
        -------
        popt_U0 : array float
            Fitted parameter, U0 (eV). Return from scipy.curve_fit.
        pcov_U0 : 2D array
            The estimated approximate covariance of popt_U0. Return from scipy.curve_fit..
        perr_U0 : float
            One standard deviation errors on the parameters (eV). 
            perr_U0 = np.sqrt(np.diag(pcov_U0))

        """
        popt_U0, pcov_U0 = curve_fit(self._Fermi_rule_fit, band_energy, band_width, 
                                     p0=[intial_guess_u0], bounds=fitting_bounds_u0)
        perr_U0 = np.sqrt(np.diag(pcov_U0))
        return popt_U0, pcov_U0, perr_U0