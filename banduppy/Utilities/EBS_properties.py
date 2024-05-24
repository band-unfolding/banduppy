import numpy as np
from ..BasicFunctions.general_functions import SaveData2File, _draw_line_length

### ===========================================================================
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
        k_labels=[(kl[b1:b2].mean(),"/".join(set(ll[b1:b2]))) for b1,b2 in zip(borders,borders[1:])]
        
        special_kpts_labels = [label[1] for label in k_labels]
        special_kpts_poss = [label[0] for label in k_labels]
        return special_kpts_labels, special_kpts_poss

class BandCentersBroadening:
    """
    Find band centers and broadening of the unfolded band structure. 
    The implementation is based on the SCF algorithm of automatic band center 
    determination from PRB 89, 041407(R) (2014) paper.
    Original implementation: 
        https://github.com/band-unfolding/bandup/utils/post_unfolding/
        locate_band_centers_and_estimate_broadening/find_band_centers_and_broadenings.py

    """
    def __init__(self, unfolded_bandstructure, min_dN_pre_screening:float=1e-4,
                 threshold_dN_2b_trial_band_center:float=0.05,
                 min_sum_dNs_for_a_band:float=0.05, 
                 precision_pos_band_centers:float=1e-5,
                 err_tolerance_compare_kpts:float=1e-8,
                 print_log='low'):
        """
        Intializing BandCentersBroadening class. Play with the different cut-offs
        and threshold values here.

        Parameters
        ----------
        unfolded_bandstructure : numpy array
            Unfolded effective band structure. 
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
        min_dN_pre_screening : float, optional
            Discard the bands which has weights below min_dN_pre_screening. This
            pre-screening step helps to minimize the data that will processed
            now on. The default is 1e-4.
        threshold_dN_2b_trial_band_center : float, optional
            Initial guess of the band centers based on the threshold wights. 
            The default is 0.05.
        min_sum_dNs_for_a_band : float, optional
            Cut off criteria for minimum weights that band center should have. 
            The band centers with lower weights than min_sum_dNs_for_a_band will be
            discarded during SCF refinements. If min_sum_dNs_for_a_band  
            is smaller than threshold_dN_2b_trial_band_center, min_sum_dNs_for_a_band
            will be reset to threshold_dN_2b_trial_band_center value.
            The default is 0.05.
        precision_pos_band_centers : float, optional (in eV)
            Precision when compared band centers from previous and latest SCF
            iteration. SCF is considered converged if this precision is reached.
            The default is 1e-5.
        err_tolerance_compare_kpts : float, optional
            The tolerance to group the bands set per unique kpoints. 
            The default is 1e-8.
        print_log : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """
        # Pre-screen band structure data to minimize unnecessary small weights bands
        self.unfolded_bandstructure_ = self._pre_screen_bandstr_data(unfolded_bandstructure, 
                                                                     min_dN_pre_screening)
        
        # Setting parameters
        self.prec_pos_band_centers = precision_pos_band_centers
        self.err_tolerance = err_tolerance_compare_kpts
        self.print_output = print_log
        
        # Reset some parameters based on condition check
        self.min_sum_dNs_for_each_band, self.threshold_dN_trial_band_center = \
            self._check_min_sum_dNs_condition(min_sum_dNs_for_a_band, 
                                              threshold_dN_2b_trial_band_center)
        
    def scfs_band_centers_broadening(self, collect_data_scf:bool=False,
                                     save_data = {'save2file': False, 
                                                  'fdir': '.',
                                                  'fname': 'unfolded_bandcenters',
                                                  'fname_suffix': ''}):
        """
        Find the band centers and broadening for a band structure. 
        The implementation is based on the SCF algorithm of automatic band center 
        determination from PRB 89, 041407(R) (2014) paper.
        Original implementation: 
            https://github.com/band-unfolding/bandup/utils/post_unfolding/
            locate_band_centers_and_estimate_broadening/find_band_centers_and_broadenings.py

        Parameters
        ----------
        collect_data_scf : bool, optional
            Whether to save the dtails of band centers in each SCF cycles. 
            The default is False.
        save_data : dictionary, optional
            save2file :: Save unfolded kpoints or not? 
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
            Format: kpoint coordinate, Band center, Band width, Sum of dN.
        gather_scf_data_ : dictionary of dictionary of array or None
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_data_scf is false.

        """
        print(f"{'-'*72}\n- Finding band center and broadening for band structure...")
        self.unfolded_bandstructure_ = self._eliminate_duplicate_kpts_list(self.unfolded_bandstructure_)
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
        
        save_data = SaveData2File.default_save_settings(save_data)
        if save_data['save2file']:
            self._save_band_centers(data2save=final_results, 
                                    save_dir=save_data["fdir"], 
                                    file_name=save_data["fname"], 
                                    file_name_suffix=save_data["fname_suffix"])       
        return final_results, gather_scf_data_
        
    def _scfs_band_centers_broadening_kpt(self, unfolded_bandstructure_current_kpt,
                                          collect_data_scf:bool=False):
        """
        Find the band centers and broadening for a particular kpoint.

        Parameters
        ----------
        unfolded_bandstructure_current_kpt : 2d numpy array
            Unfolded effective band structure. 
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
        collect_data_scf : bool, optional
            Whether to save the dtails of band centers in each SCF cycles. 
            The default is False.

        Returns
        -------
        k_index_ : int
            Index of the kpoint.
        guess_band_details : 2d numpy array
            Band center details at the particular kpoint.
            Format: kpoint coordinate, Band center, Band width, Sum of dN.
        all_data_ : dict or None
            Dictionary containing the band center details of all SCF cycles . 
            Return None if collect_data_scf is False.

        """
        # Collect kpoint information
        kpoints_cord = unfolded_bandstructure_current_kpt[0, :2]
        k_index_ = int(kpoints_cord[0])
        print (f"{'-'*72}\n- Finding band center for kpoint: {k_index_}")
        all_data_ = {} if collect_data_scf else None
        
        # Collect dNs and energies in array
        dNs_for_current_kpt = unfolded_bandstructure_current_kpt[:, -1]
        energies_for_current_kpt = unfolded_bandstructure_current_kpt[:, 2]
        min_energy, max_energy = min(energies_for_current_kpt), max(energies_for_current_kpt)
        
        # Initialize gussed band centers
        ## Apply threshold dN for trial band center
        guess_band_centers = energies_for_current_kpt[dNs_for_current_kpt >= 
                                                      self.threshold_dN_trial_band_center]
        n_guesses_bc_start = len(guess_band_centers)

        # Run self-consistence loop
        count = 0
        converged = False
        while(not converged):
            count += 1
            if self.print_output == 'high': print(f'-- SCF cycle: {count}')
            guess_band_details = self._calculate_guess_band_details(guess_band_centers, 
                                                                    min_energy, max_energy,
                                                                    energies_for_current_kpt, 
                                                                    dNs_for_current_kpt)
            refined_band_centers = self._refine_band_centers(guess_band_details, min_energy, 
                                                             self.min_sum_dNs_for_each_band)
            converged = self._check_convergence(guess_band_centers, refined_band_centers, 
                                                self.prec_pos_band_centers)
            if converged:
                guess_band_details = self._calculate_guess_band_details(refined_band_centers, 
                                                                        min_energy, max_energy,
                                                                        energies_for_current_kpt, 
                                                                        dNs_for_current_kpt)
                guess_band_details = guess_band_details[guess_band_details[:, -1] >= 
                                                        self.min_sum_dNs_for_each_band]
                guess_band_details = np.insert(guess_band_details, 0, kpoints_cord[1], axis=1)
                n_guesses_bc_end = len(guess_band_details)
                if self.print_output == 'high':
                    print('-- Positions of the band centers converged:')
                    print(f'\t--- Precision reached: {1000.0 * self.prec_pos_band_centers} meV')
                    print(f'\t--- Total SCF steps: {count}')
                    print(f'\t--- Start number of band centers: {n_guesses_bc_start}')
                    print(f'\t--- Final number of band centers: {n_guesses_bc_end}')
            else:
                guess_band_centers = refined_band_centers
                
            if collect_data_scf: all_data_[count] = guess_band_details
        # guess_band_details = (#kpoint coordinate #Band center #Band width #Sum of dN)
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
            Initial guess of the band centers based on the threshold wights.

        Returns
        -------
        min_sum_dNs_for_a_band : float
            Cut off criteria for minimum weights that band center should have. 
            The band centers with lower weights than min_sum_dNs_for_a_band will be
            discarded during SCF refinements.
        threshold_dN_2b_trial_band_center : float
            Initial guess of the band centers based on the threshold wights.

        """
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
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
        min_dN : float
            Discard the bands which has weights below min_dN_pre_screening. This
            pre-screening step helps to minimize the data that will processed
            now on.

        Returns
        -------
        2d numpy array
            Unfolded effective band structure after removing small weights centers. 
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.

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
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.

        Returns
        -------
        unfolded_bandstructure : 2d numpy array
            Unfolded effective band structure after removing duplicates. 
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.

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

    def _group_unique_kpts_data(cls, unfolded_bandstructure, err_tolerance:float=1e-8):
        """
        Group bands at each unique k-points.

        Parameters
        ----------
        unfolded_bandstructure : 2d numpy array
            Unfolded effective band structure. 
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
        err_tolerance : float, optional
            The tolerance to group the bands set per unique kpoints. 
            The default is 1e-8.

        Returns
        -------
        list
            List of unfolded effective band structure group by kpoints.
            Format: k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor..

        """
        # Get unique kpoints coordinate
        unique_kpts_coords = np.unique(unfolded_bandstructure[:, 1])
        # Group the rows based on unique values in the specified column
        return [unfolded_bandstructure[abs(unfolded_bandstructure[:, 1] - val) < err_tolerance] 
                for val in unique_kpts_coords]
    
    @classmethod
    def _weighted_avg_and_std(cls, values, weights):
        """
        Calculate average and variance of list of values. 

        Parameters
        ----------
        values : 1d array or list
            Values to average.
        weights : 1d array or list
            Values to use as weights in averaging.

        Returns
        -------
        tuple (float, float)
            (average value, std deviation).

        """
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights) 
        return (average, np.sqrt(variance))
    
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

    @classmethod
    def _calculate_guess_band_details(cls, guess_band_centers, min_energy, max_energy,
                                      energies_, dNs_):
        """
        Calculates band ceneters, band weights, and band width.

        Parameters
        ----------
        guess_band_centers : float array
            List of guessed/refined band ceneters.
        min_energy : float
            Minimum energy.
        max_energy : float
            Maximum energy.
        energies_ : float array
            All bands energies at specific kpoint.
        dNs_ : float array
            All band weights at specific kpoint.

        Returns
        -------
        2d numpy array
            Band center details at the particular kpoint.
            Format: kpoint coordinate, Band center, Band width, Sum of dN.

        """
        guess_energy_width = cls._calculate_possible_energy_width(guess_band_centers, 
                                                                  min_energy, max_energy)
        guess_band_details = [] 
        for iband in range(len(guess_band_centers)):
            indices_of_enegies_spread_in_band = \
                np.argwhere((energies_ >= guess_energy_width[iband][0]) & 
                            (energies_ < guess_energy_width[iband][1])).flatten()       
            band_weight_ = np.sum(dNs_[indices_of_enegies_spread_in_band])
            band_centers_, band_width_ = \
                cls._weighted_avg_and_std(values=energies_[indices_of_enegies_spread_in_band], 
                                          weights=dNs_[indices_of_enegies_spread_in_band])
            # Band center, standard width, weight
            guess_band_details.append([band_centers_, band_width_, band_weight_])
            
        return np.array(guess_band_details)

    @classmethod
    def _refine_band_centers(cls, guess_band_details, min_energy, min_sum_dNs_for_a_band):
        """
        Discard band centers which are too close in energy. Reducing the number 
        of too close energy values.
        
        Discard band centers with
        abs(e_n,current - e_(n-1),current) < 2*max[band_n_width, band_n-1_width]

        Parameters
        ----------
        guess_band_details : 2d numpy array
            Band center details at the particular kpoint.
            Format: kpoint coordinate, Band center, Band width, Sum of dN.
        min_energy : TYPE
            DESCRIPTION.
        min_sum_dNs_for_a_band : float
            Cut off criteria for minimum weights that a band center should have. 
            The band centers with lower weights than min_sum_dNs_for_a_band will be
            discarded during SCF refinements.

        Returns
        -------
        2d numpy array
            Band center details at the particular kpoint after refining.
            Format: kpoint coordinate, Band center, Band width, Sum of dN.

        """
        refined_band_centers = []
        for iband in range(len(guess_band_details)):
            bc, bwidth, b_weight = guess_band_details[iband]
            previous_bc, previous_bwidth, previous_b_weight = guess_band_details[iband-1]
            #print(iband, bc, bwidth, b_weight, previous_bc, previous_bwidth, previous_b_weight)
            valid_bc = False
            if(b_weight < min_sum_dNs_for_a_band or \
               abs(bc - previous_bc) < 2.0 * max([bwidth, previous_bwidth])): 
                try:
                    if(abs(b_weight / previous_b_weight) > 1.0):
                        del refined_band_centers[-1]
                        valid_bc = True
                except:
                    pass
            else:
                valid_bc = True

            if(valid_bc):
                refined_band_centers.append(bc)
        return np.array(refined_band_centers)

    @staticmethod
    def _check_convergence(old_band_centers, new_band_centers, prec_pos_band_centers):
        """
        Check if the two band centers are close. If true, then the band ceneter
        is considered converged in SCF.

        Parameters
        ----------
        old_band_centers : 1d numpy array
            Previous band centers.
        new_band_centers : 1d numpy array
            Refined band centers.
        prec_pos_band_centers : float
            Precision when compared band centers from previous and latest SCF
            iteration. SCF is considered converged if this precision is reached.

        Returns
        -------
        bool
            If two arrays are same or not.

        """
        return all(abs(old_band_centers[:len(new_band_centers)] - new_band_centers) 
                   <= prec_pos_band_centers)

    def _save_band_centers(self, data2save, save_dir, file_name, file_name_suffix):
        """
        Save unfolded band centers data.
        Format: kpoint coordinate, Band center, Band width, Sum of dN.

        Parameters
        ----------
        data2save : 2d array
            Data to save.
        save_dir : str or path
            Directory path where to save the file.
        file_name : str
            Name of the file.
        file_name_suffix : str
            Suffix to add to the file name.

        Returns
        -------
        None.

        """
        if self.print_output is not None: 
            print(f"{'='*_draw_line_length}\n- Saving unfolded band centers to file...")
        header_msg  = " Unfolded band centers data\n"
        header_msg += " k on path (A^-1), Band center energy, Band width, Sum of dN\n"
        # Save the sc-kpoints in file
        save_f_name = \
        SaveData2File.save_2_file(data=data2save, 
                                  save_dir=save_dir, file_name=file_name,
                                  file_name_suffix=f'{file_name_suffix}.dat', 
                                  header_txt=header_msg, comments_symbol='#',
                                  print_log=bool(self.print_output))
        if self.print_output is not None: 
            print(f'-- Filepath: {save_f_name}\n- Done')