from .unfolding import Properties
from .BasicFunctions import _SaveData2File
from . import __version__
import numpy as np

### =========================================================================== 
class HighLevelProperties(Properties):
    """
    Calculate properties from unfolded band structure. This class
    is introduced to define high-level finctions that combines multiple small 
    calculation together. E.g., Instead of seperate functions calling for 
    individual parabolic and hyperbolic effective mass calculation, one can call
    a singfle function calculate_effective_masses() from this class to calculate
    both the effective masses in a single shot.

    """
    def __init__(self, print_log='low'):
        """
        Initialize the BandUPpy Properties class.

        Parameters
        ----------
        print_log : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """       
        Properties.__init__(self, print_log=print_log)
        
    # Initital screening: fit data points with band energies below 1 eV
    def _screen_function_band_centers(self, band_centers_data_process, 
                                      inital_guess_param):
        ### Using hyperbolic of unstrained structures theoretical 
        ### liner interpolated effective mass
        EnergyFit = self.effective_mass_fit_functions(band_centers_data_process[:, 1], 
                                                      inital_guess_param,
                                                      parabolic_dispersion=False,
                                                      hyperbolic_dispersion_positive=
                                                      self.hyperbolic_dispersion_positive,
                                                      hyperbolic_dispersion_negative=
                                                      self.hyperbolic_dispersion_negative)
        # Step 1: Calculate residuals (differences)
        residuals = EnergyFit - band_centers_data_process[:, 2]
        if self.use_residual_std_dev:
            # Step 2: Calculate the standard deviation of residuals
            cut_off_residual = self.residual_std_dev_start * np.std(residuals)
        else:
            cut_off_residual = self.residual_eV_cutoff
            
        # Step 3: Find the indices where the residuals are less than cut-off
        #inlier_indices = np.argwhere(np.abs(residuals) < cut_off_residual).flatten() 
        
        return np.argwhere(np.abs(residuals) < cut_off_residual).flatten() 
    
    def _screen_band_centers(self, band_centers_data_process):
        if self.log_scf_info:
            print('-- Refining band center fitting for effective mass and alloy scattering') 
        TotdalData = len(band_centers_data_process)
        # In scipy curve_fit chisq=sum((r/sigma)**2) => 1/sigma is the weights
        inlier_indices = self._screen_function_band_centers(band_centers_data_process, 
                                                            self.inital_guess_param)  
        band_centers_data_process_ = band_centers_data_process[inlier_indices]
        inlier_count_screen = len(inlier_indices)
        # Rectifying screening
        popt_check = self.inital_guess_param
        ii = 1
        while True:
            if self.log_scf_info: 
                print(f'--- Cycle-{ii:<3}: Number of outliers = {TotdalData - inlier_count_screen}')
            # calculate effective mass using processed data: excluding outliers
            _, popt_check, _, _ = self.calculate_effecfive_mass(band_centers_data_process_[:, 1], 
                                                                band_centers_data_process_[:, 2],
                                                                ignore_kshift_cbm_fit=
                                                                self.ignore_kshift_cbm_fit,
                                                                fit_weights=
                                                                1/band_centers_data_process_[:, -1], 
                                                                absolute_weights=self.absolute_weights,
                                                                initial_guess_params=popt_check,
                                                                parabolic_dispersion=False,
                                                                hyperbolic_dispersion_positive=
                                                                self.hyperbolic_dispersion_positive,
                                                                hyperbolic_dispersion_negative=
                                                                self.hyperbolic_dispersion_negative)
            # Fit the hyperbolic function for all the data
            inlier_indices = self._screen_function_band_centers(band_centers_data_process, 
                                                                popt_check) 
            # Break SCF loop condition
            inlier_count_screen, inlier_count_start = len(inlier_indices), len(band_centers_data_process_)
            if inlier_count_screen == inlier_count_start:
                if self.log_scf_info: print('--- Cycle-{ii:<3}: SCF converged.')
                break
            elif ii > self.max_scf_loop:
                if self.log_scf_info: print(f'--- Warning: Maximum scf loop ({ii}) iteratation is reached. SCF not converged.')
                break
            else:
                band_centers_data_process_ = band_centers_data_process[inlier_indices]
                ii+=1
                self.residual_std_dev_start -= 1 
                if self.residual_std_dev_start<1: self.residual_std_dev_start=1

        return inlier_indices, popt_check

    def calculate_effective_masses(self, band_centers_data_process, 
                                   initial_guess_params_parabolic,
                                   initial_guess_params_hyperbolic,
                                   ignore_kshift_cbm_fit:bool=True,
                                   absolute_weights:bool=True,
                                   hyperbolic_dispersion_positive:bool=True,
                                   hyperbolic_dispersion_negative:bool=False):
        """
       High-level function. This function calculates both the parabolic and hyperbolic 
       effective masses.

        Parameters
        ----------
        band_centers_data_process : TYPE
            DESCRIPTION.
        initial_guess_params_parabolic : array_like, optional
            Initial guess for the parabolic fitting parameters (length N). If None,  
            then initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            ValueError is raised).
        initial_guess_params_hyperbolic : array_like, optional
            Initial guess for the hyperbolic fitting parameters (length N). If None,  
            then initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            ValueError is raised).
        ignore_kshift_cbm_fit : bool, optional
            Ignore fitting kshift and cbm parameters during fitting. The default is True.
        absolute_weights : bool, optional
            If True, fit_weights is used in an absolute sense and the estimated parameter 
            covariance pcov reflects these absolute values.
            If False, only the relative magnitudes of the fit_weights values matter. 
            The default is True. 
        hyperbolic_dispersion_positive : bool, optional
            Fit hyperbolic model (upward hyperbola) of band dispersion. The default is False.
        hyperbolic_dispersion_negative : bool, optional
            Fit hyperbolic model (downward hyperbola) of band dispersion. The default is False.

        Returns
        -------
        m_star_p_results : tuple, results for parabolic dispersion
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
        m_star_h_results : tuple, results for hyperbolic dispersion
            m_star : (float, float)
                Calculated effective mass (m_star[0]) and error (m_star[1]) in m_0 unit.
            popt : array
                Optimal values for the parameters so that the sum of the squared
                residuals of ``f(xdata, *popt) - ydata`` is minimized.
            pcov : 2-D array
                The estimated approximate covariance of popt. 
            params_errors : 1-D array
                One standard deviation error in parameters.
                perr = np.sqrt(np.diag(pcov)).
        """
        
        # In scipy curve_fit chisq=sum((r/sigma)**2) => 1/sigma is the weights  
        weightings_ = 1/band_centers_data_process[:, -1] 
        ## ====================================================================================================================
        # Get effective mass for parabolic dispersion
        # m_star_p_results = (m_star_p, popt_p, pcov_p, params_error_p)
        m_star_p_results = self.calculate_effecfive_mass(band_centers_data_process[:, 1],
                                                        band_centers_data_process[:, 2],
                                                        initial_guess_params=
                                                        initial_guess_params_parabolic,
                                                        ignore_kshift_cbm_fit=
                                                        ignore_kshift_cbm_fit,
                                                        fit_weights=weightings_, 
                                                        absolute_weights=absolute_weights,
                                                        parabolic_dispersion=True)
        ## ====================================================================================================================
        # Get effective mass for hyperbolic dispersion
        # m_star_h_results = (m_star_h, popt_h, pcov_h, params_error_h)
        m_star_h_results = self.calculate_effecfive_mass(band_centers_data_process[:, 1],
                                                        band_centers_data_process[:, 2], 
                                                        initial_guess_params=
                                                        initial_guess_params_hyperbolic,
                                                        ignore_kshift_cbm_fit=
                                                        ignore_kshift_cbm_fit,
                                                        fit_weights=weightings_, 
                                                        absolute_weights=absolute_weights,
                                                        parabolic_dispersion=False,
                                                        hyperbolic_dispersion_positive=
                                                        hyperbolic_dispersion_positive,
                                                        hyperbolic_dispersion_negative=
                                                        hyperbolic_dispersion_negative,)
        return m_star_p_results, m_star_h_results
        
    def scf_refine_effecfive_mass_calculator(self, band_centers_data, 
                                             initial_guess_params=None,
                                             residual_cutoff_in_eV:float=1.0,
                                             use_residual_std_dev:bool=False,
                                             residual_std_dev_start:float=1.0,
                                             initial_guess_fit_dispersion = 'positive_hyperbolic_dispersion',
                                             ignore_kshift_cbm_fit:bool=True, 
                                             kshift:float=None, absolute_weights:bool=False,
                                             max_scf_loop:int=100):
        self.residual_eV_cutoff = residual_cutoff_in_eV
        self.use_residual_std_dev = use_residual_std_dev
        self.residual_std_dev_start = residual_std_dev_start
        self.ignore_kshift_cbm_fit = ignore_kshift_cbm_fit
        self.absolute_weights = absolute_weights
        self.max_scf_loop = max_scf_loop
        self.inital_guess_param = initial_guess_params
        self.log_scf_info = False
        if self.print_log_info is not None:
            if self.print_log_info in ['medium', 'high']:
                self.log_scf_info = True
        #----------------------------------------------------------------------
        band_centers_data_process = band_centers_data.copy()
        kshift_ = band_centers_data_process[0,1] if kshift is None else kshift
        band_centers_data_process[:, 1] -= kshift_
        #----------------------------------------------------------------------
        self.hyperbolic_dispersion_positive, self.hyperbolic_dispersion_negative = True, False
        if initial_guess_fit_dispersion == 'positive_hyperbolic_dispersion':
            self.hyperbolic_dispersion_positive = True  
        else:
            self.hyperbolic_dispersion_positive = False
            self.hyperbolic_dispersion_negative = True
        #----------------------------------------------------------------------     
        # Refine the band centers with scf removal: 
        # get the band centers indices that will be used for 
        # further fitting in the next step.
        inlier_indices, popt_check = self._screen_band_centers(band_centers_data_process)
        #----------------------------------------------------------------------
        # Calculate the parabolic and hyperbolic effective masses
        m_star_p_results, m_star_h_results =\
        self.calculate_effective_masses(band_centers_data_process[inlier_indices],
                                        popt_check, popt_check,
                                        ignore_kshift_cbm_fit=self.ignore_kshift_cbm_fit,
                                        absolute_weights=self.absolute_weights,
                                        hyperbolic_dispersion_positive=
                                        self.hyperbolic_dispersion_positive,
                                        hyperbolic_dispersion_negative=
                                        self.hyperbolic_dispersion_negative)
        return inlier_indices, m_star_p_results, m_star_h_results
    
    def merge_zero_weight_2_sc_kpoint_file(self, nonzero_weight_kp_file, 
                                           zero_weight_kp_file, 
                                           save_dir='.', file_name:str='KPOINTS_merged', 
                                           file_name_suffix:str='', 
                                           file_format:str = 'vasp'):
        with open(nonzero_weight_kp_file, 'r') as f:
            lines = []
            for i in range(3): 
                lines.append(f.readline())

        if not lines[-1].lower().startswith('rec'):
            raise NotImplementedError("Only reciprocal mode Kpoints in IBZKPT file merginf has been implemented so far.")
          
        ibz_kps = np.genfromtxt(nonzero_weight_kp_file, skip_header=3, max_rows=int(lines[1]), comments='#')
        zero_weight_sc_kps = np.genfromtxt(zero_weight_kp_file, skip_header=3, comments='!')
        merge_data = np.concatenate((ibz_kps, zero_weight_sc_kps), axis=0)
        
        header_msg  = f"K-points (IBZKPT plus zero weighted KPOINTS_SC) generated using banduppy-{__version__} package"
        header_msg += f"\n{len(merge_data)}\nreciprocal"
        _SaveData2File._save_sc_kpts_2_file(data=merge_data,
                                            save_dir=save_dir, file_name=file_name,
                                            file_name_suffix=file_name_suffix, 
                                            file_format=file_format,
                                            header_txt=header_msg, 
                                            footer_txt='',
                                            print_log=self.print_log_info,
                                            print_msg='Saving merged KPOINT file')