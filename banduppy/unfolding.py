import numpy as np
from .src import _BandFolding, _BandUnfolding, _GeneralFnsDefs
from .Utilities import _GeneralFunctionsDefs, _EBSplot, _FoldingDegreePlot, _BandCentersBroadening, _EffectiveMass

### ===========================================================================    
class Unfolding(_BandFolding, _BandUnfolding, _EBSplot, _FoldingDegreePlot):
    """
    Band folding from primitive to supercell.

    """
    def __init__(self, supercell=None, print_log='low'):
        """
        Initialize the BandUPpy Unfolding class.

        Parameters
        ----------
        supercell : 3X3 matrix, optional
            Primitive-to-supercell transformation matrix. The default is Identity matrix.
        print_log : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """       
        if print_log is not None: print_log = print_log.lower()
        _BandFolding.__init__(self, supercell=supercell, print_info=print_log)
        
    def propose_maximum_minimum_folding(self, pathPBZ, min_num_pts:int=5, max_num_pts:int=20,
                                        serach_mode:str='brute_force', draw_plots:bool=True, 
                                        save_plot:bool=False, save_dir='.', save_file_name=None,  
                                        CountFig=None, yaxis_label:str='Folding degree (%)',
                                        xaxis_label:str='number of kpoints', line_color='k'):
        """
        Calculates SC Kpoints from PC kpoints and returns percent of folding.
        Maximum and Minimum degree of folding are reported.
        
        Folding percent = ((#of PC kpoints - #of folded SC Kpoints)/(#of PC kpoints))*100

        Parameters
        ----------
        pathPBZ : ndarray/list
            PC kpoint path nodes in reduced coordinates.
        min_num_pts : int, optional
            Minimum number of kpoints division in the k-path. The default is 5.
        max_num_pts : int, optional
            Maximum number of kpoints division in the k-path. The default is 20.
        serach_mode : ['brute_force'], optional
            Method to calculate SC Kpoints. The default is 'brute_force'.
        draw_plots : bool, optional
            Plot folding vs number of k-points. The default is True.
            If True, also returns fig, ax, and CountFig.
        save_plot : bool, optional
            Save plots or not. The default is False.
        save_dir : str/path, optional
            Directory where to save the plots. The default is '.'.
        save_file_name : str, optional
            Name of the file to ba saved. The default is None. If None, figure
            is not saved.
        CountFig : int, optional
            Figure count. The default is None. If None, nothing is is done. Else,
            returns CountFig increased by 1.
        yaxis_label : str, optional
            yaxis label. The default is 'Folding degree (%)'.
        xaxis_label : str, optional
            xaxis label. The default is 'number of kpoints'.
        line_color : matplotlib color, optional
            Line color. The default is 'k'.

        Returns
        -------
        proposed_folding_results : dictionary
            {index: ((start node, end node), folding data)}
            index : Index of path segment searched from the pathPBZ list supplied.
            folding data : 2d array with each row containing number of division in the 1st
            column and percent of folding in the 2nd column.
            
            If draw_plots=True, also returns fig, ax, and CountFig.
        """
        proposed_folding_results = \
             self._propose_best_least_folding(pathPBZ, min_num_pts=min_num_pts, 
                                              max_num_pts=max_num_pts,
                                              serach_mode=serach_mode)
        if draw_plots:
            _FoldingDegreePlot.__init__(self, fold_results_dictionary=proposed_folding_results, 
                                        save_figure_dir=save_dir)
            fig, ax, CountFig = self._plot_folding(save_file_name=save_file_name, 
                                                   CountFig=CountFig, 
                                                   yaxis_label=yaxis_label,
                                                   xaxis_label=xaxis_label, 
                                                   line_color=line_color)
            return proposed_folding_results, fig, ax, CountFig
        return proposed_folding_results
        
    def generate_SC_Kpts_from_pc_k_path(self, pathPBZ=None, nk=11, labels=None, kpts_weights=None, 
                                        save_all_kpts:bool=False, save_sc_kpts:bool=False, 
                                        save_dir='.', file_name:str='', 
                                        file_name_suffix:str='', file_format:str='vasp'):
        """
        Generate supercell kpoints from reference primitive BZ k-path.

        Parameters
        ----------
        pathPBZ : ndarray/list, optional
            PC kpoint path nodes in reduced coordinates. 
            If the segmant is skipped, put a None between nodes.
            E.g. [[1/2,1/2,1/2], [0,0,0],None, [3/8,3/8,3/4], [0,0,0]] for [LGKG]
            The default is None. 
        nk : int ot tuple, optional
            Number of kpoints in each k-path segment. The default is 11.
            None in pathPBZ is not part of the segments. 
            E.g. In [[1/2,1/2,1/2], [0,0,0],None, [3/8,3/8,3/4], [0,0,0]] there are
            only 2 segments.
        labels : string ot list of strings
            Labels of special k-points, either as a continuous list or string. 
            Do not use ',' or multidimentional list do define disjoint segmants.
            e.g. Do not use labels='LG,KG'. Use labels='LGKG'. The 'None' in the
            pathPBZ will take care of the disjoint segments.
            If multiple word needs to be single label, use list.
            e.g. labels=['X','Gamma','L']. Do not use string labels='XGammaL'.
            The default is None. If None, the special
            kpoints will be indexed as 1,2,3,...
        kpts_weights : int or float or 1d numpy array, optional
            Weights of the SC kpoints. The default is None. If none, no weights are padded
            in the generated SC K-points list.
        save_all_kpts : bool, optional
            Save the PC kpoints, generated SC kpoints, and SC-PC kpoints mapping. 
            The default is False. If True, has precedence over save_sc_kpts.
        save_sc_kpts : bool, optional
            Save the generated SC kpoints. The default is False.
        save_dir : str/path_object, optional
            Directory to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The default is ''.
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>
        file_name_suffix : str, optional
            Suffix to add after the file_name. The default is ''.
        file_format : ['vasp'], optional
            Format of the file. The default is 'vasp'. 
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>

        Returns
        -------
        ndarray of floats
            PC kpoints list.
        ndarray of floats
            SC kpoints list.
        ndarray of int
            PC unique kpoints indices for reverse engineer.
        ndarray of int
            SC unique kpoints indices for reverse engineer.

        """
        return self._generate_SC_K_from_pc_k_path(pathPBZ=pathPBZ, nk=nk, labels=labels, 
                                                  kpts_weights=kpts_weights, 
                                                  save_all_kpts=save_all_kpts, 
                                                  save_sc_kpts=save_sc_kpts, 
                                                  save_dir=save_dir, file_name=file_name, 
                                                  file_name_suffix=file_name_suffix, 
                                                  file_format=file_format)
            
    def generate_SC_Kpts_from_pc_kpts(self, kpointsPBZ=None, kpts_weights=None,
                                      save_all_kpts:bool=False, save_sc_kpts:bool=False, 
                                      save_dir='.', file_name:str='', 
                                      file_name_suffix:str='', file_format:str='vasp', footer_msg=None,
                                      special_kpoints_pos_labels=None):
        """
        Generate supercell kpoints from reference primitive kpoints.

        Parameters
        ----------
        kpointsPBZ : ndarray, optional
            PC kpoint list. The default is None. 
        kpts_weights : int or float or 1d numpy array, optional
            Weights of the SC kpoints. The default is None. If none, no weights are padded
            in the generated SC K-points list.
        save_all_kpts : bool, optional
            Save the PC kpoints, generated SC kpoints, and SC-PC kpoints mapping. 
            The default is False. If True, has precedence over save_sc_kpts.
        save_sc_kpts : bool, optional
            Save the generated SC kpoints. The default is False.
        save_dir : str/path_object, optional
            Directory to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The default is ''.
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>
        file_name_suffix : str, optional
            Suffix to add after the file_name. The default is ''.
        file_format : ['vasp'], optional
            Format of the file. The default is 'vasp'. 
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>
        footer_msg : str, optional
            String that will be written at the end of the file. The default is PC kpoints list.
        special_kpoints_pos_labels : dictionary, optional
            Special kpoints position_index in PC kpoints list as key and label as value. 
            Will be used in plotting. Default is None.

        Returns
        -------
        ndarray of floats
            PC kpoints list (orginal provided k-path, full list).
        ndarray of floats
            PC kpoints list (unique).
        ndarray of floats
            SC kpoints list (unique).
        dictionary of int/list
            Mapping of SC kpts (K), PC unique kpts (k unique), and PC full kpts (k) indices.
            format: {K index: K -> k index unique: k unique -> k index: k}
            This mapping can be used for reverse engineer latter.
        dictionary or None
            Position and labels of special kpoints, will be used in plotting.

        """
        return self._generate_K_from_k(kpointsPBZ=kpointsPBZ, kpts_weights=kpts_weights,
                                       save_all_kpts=save_all_kpts, 
                                       save_sc_kpts=save_sc_kpts, 
                                       save_dir=save_dir, file_name=file_name, 
                                       file_name_suffix=file_name_suffix, 
                                       file_format=file_format, footer_msg=footer_msg,
                                       special_kpoints_pos_labels=special_kpoints_pos_labels)
    
    def Unfold(self, bandstructure,  
               PBZ_kpts_list_full=None, SBZ_kpts_list=None, SBZ_PBZ_kpts_map=None,
               kline_discontinuity_threshold = 0.1,
               save_unfolded_kpts = {'save2file': False, 
                                     'fdir': '.',
                                     'fname': 'kpoints_unfolded',
                                     'fname_suffix': ''},
               save_unfolded_bandstr = {'save2file': False, 
                                        'fdir': '.',
                                        'fname': 'bandstructure_unfolded',
                                        'fname_suffix': ''}):
        """
        Unfold the band structure.

        Parameters
        ----------
        bandstructure : irrep.bandstructure.BandStructure
            irrep.bandstructure.BandStructure.
        PBZ_kpts_list_full : ndarray, optional
            List of PC k-points. If None, try to find the list from class instance.
            The default is None.
        SBZ_kpts_list : ndarray, optional
            List of SC K-points. If None, try to find the list from class instance.
            The default is None.
        SBZ_PBZ_kpts_map : dictionary, optional
            Mapping of SC generated K-points indices and PC k-points indices. 
            If None, try to find the list from class instance.
            The default is None.
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
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor]
        numpy ndarray
            Unfolded effective band structure k-path.
            Format: [k on path (A^-1)]

        """
        if (PBZ_kpts_list_full is None) or \
            (SBZ_kpts_list in None) or (SBZ_PBZ_kpts_map is None):
            _BandUnfolding.__init__(self, self.transformation_matrix, 
                                    self.PBZ_kpts_list_org, self.SBZ_kpts_list, 
                                    self.SBZ_PBZ_kpts_mapping,
                                    print_info=self.print_information)
        else:
            _BandUnfolding.__init__(self, self.transformation_matrix, 
                                    PBZ_kpts_list_full, SBZ_kpts_list, 
                                    SBZ_PBZ_kpts_map, 
                                    print_info=self.print_information)
        
        return self._unfold(bandstructure, 
                            kline_discontinuity_threshold = kline_discontinuity_threshold,
                            save_unfolded_kpts = save_unfolded_kpts,
                            save_unfolded_bandstr = save_unfolded_bandstr)

    def plot_ebs(self, fig=None, ax=None, save_figure_dir='.', save_file_name=None,  
                 CountFig=None, Ef=None, Emin=None, Emax=None, pad_energy_scale:float=0.5, 
                 threshold_weight:float=None, mode:str="fatband", yaxis_label:str='E (eV)', 
                 special_kpoints:dict=None, plotSC:bool=True, marker='o', fatfactor=20, 
                 nE:int=100, smear:float=0.05, color='gray', color_map='viridis', 
                 show_legend:bool=True, show_colorbar:bool=False, colorbar_label:str=None, 
                 vmin=None, vmax=None, show_plot:bool=True,**kwargs_savefig):
        
        """
        Scatter/density plot of the band structure.

        Parameters
        ----------
        fig : matplotlib.pyplot figure instance, optional
            Figure instance to plot on. The default is None.
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on. If None, new figure will be created.
            The default is None.
        save_figure_dir : str, optional
            Directory where to save the figure. The default is current directory.
        save_file_name : str, optional
            Name of the figure file. If None, figure will be not saved. 
            The default is None.
        CountFig: int, optional
            Figure count. The default is None.
        Ef : float, optional
            Fermi energy. If None, set to 0.0. The default is None.
        Emin : float, optional
            Minimum in energy. The default is None.
        Emax : float, optional
            Maximum in energy. The default is None.
        pad_energy_scale: float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        threshold_weight : float, optional
            The band centers with band weights lower than the threshhold weights 
            are discarded. The default is None. If None, this is ignored.
        mode : ['fatband','density'], optional
            Mode of plot. The default is "fatband".
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plotSC : bool, optional
            Plot supercell bandstructure. The default is True.
        marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style. Marker can be either an instance of the class or 
            the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color of scatter plot of unfolded band structure. The color of supercell
            band structures is gray. The default is 'gray'.
        color_map: str/ matplotlib colormap
            Colormap for density plot. The default is viridis.
        show_legend : bool
            If show legend or not. The default is True.
        show_colorbar : bool, optional
            Plot the colorbar in the figure or not. If fig=None, this is ignored.
            The default is False.
        colorbar_label : str, optional
            Colorbar label. The default is None. If None, ignored.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers. 
            By default, the colormap covers the complete value range of the supplied data.
        show_plot : bool, optional
            To show the plot when not saved. The default is True.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure instance. If ax is not None previously generated fig instance
            will be used.
        ax : Axis instance
            Figure axis instance.
        CountFig: int or None
            Figure count.

        """
        
        _EBSplot.__init__(self, save_figure_dir=save_figure_dir)

        return self._plot(fig=fig, ax=ax, save_file_name=save_file_name, CountFig=CountFig,  
                          Ef=Ef, Emin=Emin, Emax=Emax, pad_energy_scale=pad_energy_scale, 
                          threshold_weight=threshold_weight, mode=mode, yaxis_label=yaxis_label, 
                          special_kpoints=special_kpoints, plotSC=plotSC, marker=marker, 
                          fatfactor=fatfactor, nE=nE, smear=smear, color=color, color_map=color_map,
                          show_legend=show_legend, show_colorbar=show_colorbar,
                          colorbar_label=colorbar_label, vmin=vmin, vmax=vmax, 
                          show_plot=show_plot, **kwargs_savefig)
    
class Properties(_BandCentersBroadening, _EffectiveMass):
    """
    Calculate properties from unfolded band structure.

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
        if print_log is not None: print_log = print_log.lower()
        self.print_log_info = print_log

    def collect_bandstr_data_only_in_energy_window(self, unfolded_bandstructure, Ef:float=None, 
                                                    Emin:float=None, Emax:float=None,  
                                                    pad_energy_scale:float=0.5, 
                                                    min_dN_screen:float=0.0, 
                                                    save_data = {'save2file': False, 
                                                                 'fdir': '.',
                                                                 'fname': 'unfolded_bandcenters_window',
                                                                 'fname_suffix': ''}):
        """
        Collect data within the condition and range specified. 
        Note: Returns only 1st 4 columns. Removes the spinor data part for 
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
        min_dN_screen : float, optional
            The band centers with band weights lower than the threshhold weights 
            are discarded. The default is 0.
        save_data : dictionary, optional
            save2file :: Save data to file or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name.
            The default is {'save2file': False, 'fdir': '.', 'fname': 'unfolded_bandcenters_window', 'fname_suffix': ''}.
            
         Returns
         -------
         unfolded_bandcenters_window : ndarray
             Unfolded effective band structure/band center data within the range. 

         """
         
        unfolded_bandcenters_window = \
        _GeneralFunctionsDefs._get_bandstr_data_only_in_energy_kpts_window(unfolded_bandstructure, Ef=Ef, 
                                                                           Emin=Emin, Emax=Emax,  
                                                                           pad_energy_scale=pad_energy_scale, 
                                                                           min_dN_screen=min_dN_screen)
        
        _GeneralFunctionsDefs._save_band_centers(data2save=unfolded_bandcenters_window, 
                                                 print_log=self.print_log_info,
                                                 save_data_f_prop=save_data)
        return unfolded_bandcenters_window
    
    def band_centers_broadening_bandstr(self, unfolded_bandstructure, 
                                        min_dN_pre_screening:float=1e-4,
                                        threshold_dN_2b_trial_band_center:float=0.05,
                                        min_sum_dNs_for_a_band:float=0.05, 
                                        precision_pos_band_centers:float=1e-5,
                                        err_tolerance_compare_kpts_val:float=1e-8,
                                        collect_scf_data:bool=False,
                                        save_data = {'save2file': False, 
                                                     'fdir': '.',
                                                     'fname': 'unfolded_bandcenters',
                                                     'fname_suffix': ''}):
        """
        Find band centers and broadening of the unfolded band structure.

        Parameters
        ----------
        unfolded_bandstructure : numpy array
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor]
       min_dN_pre_screening : float, optional
           Discard the bands which has weights below min_dN_pre_screening to start with. 
           This pre-screening step helps to minimize the data that will processed
           now on. The default is 1e-4. [* critical parameter]
       threshold_dN_2b_trial_band_center : float, optional
           Initial guess of the band centers based on the threshold wights. 
           The default is 0.05. [* critical parameter]
       min_sum_dNs_for_a_band : float, optional
           Cut off criteria for minimum weights that a band center should have. 
           The band centers with lower weights than min_sum_dNs_for_a_band will be
           discarded during SCF refinements. If min_sum_dNs_for_a_band  
           is smaller than threshold_dN_2b_trial_band_center, min_sum_dNs_for_a_band
           will be reset to threshold_dN_2b_trial_band_center value.
           The default is 0.05. [* critical parameter]
       precision_pos_band_centers : float, optional
           Precision when compared band centers from previous and current SCF
           iteration. SCF is considered converged if this precision is reached.
           The default is 1e-5. [not critical parameter]
       err_tolerance_compare_kpts_val : float, optional
           The tolerance to group the bands set per unique kpoints. This
           determines if two flotting point numbers are the same or not. This is not 
           a critical parameter for band center determination algorithm.
           The default is 1e-8. [not critical parameter]
        collect_scf_data : bool, optional
            Whether to save the dtails of band centers in each SCF cycles.
            The default is False.
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
        list of array
            Each array contains the final details of band centers in a particular
            kpoint. The list contains band center details for each kpoints.
            Format: [kpoint coordinate, Band center, Band width, Sum of dN]
        dictionary of dictionary of array or None
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_scf_data is false.
            Format: {kpoint_index: {SCF_cycle_index: [Band center, Band width, Sum of dN]}}

        """
        _BandCentersBroadening.__init__(self, unfolded_bandstructure=unfolded_bandstructure, 
                                        min_dN_pre_screening=min_dN_pre_screening,
                                        threshold_dN_2b_trial_band_center=threshold_dN_2b_trial_band_center,
                                        min_sum_dNs_for_a_band=min_sum_dNs_for_a_band, 
                                        precision_pos_band_centers=precision_pos_band_centers,
                                        err_tolerance_compare_kpts_val=err_tolerance_compare_kpts_val,
                                        print_log=self.print_log_info)
        return self._scfs_band_centers_broadening(collect_data_scf=collect_scf_data,
                                                  save_data=save_data)
    
    def calculate_effecfive_mass(self, kpath, band_energy, 
                                 initial_guess_params=None, 
                                 params_bounds = (-np.inf, np.inf), 
                                 fit_weights=None, absolute_weights:bool=False,
                                 parabolic_dispersion:bool=False, 
                                 hyperbolic_dispersion_positive:bool=False,
                                 hyperbolic_dispersion_negative:bool=False,
                                 params_name = ['alpha', 'kshift', 'cbm', 'gamma']):
        """
        Calculates effective mass using (hyper) parabolic band dispersion.
        
        Parameters
        ----------
        kpath : numpy array
            k on path (A^-1) of unfolded effective band structure/band centers that will
            be fitted for calculating effective mass.
        band_energy : numpy array
            Energy (in eV) of unfolded effective band structure/band centers that will
            be fitted for calculating effective mass.
        initial_guess_params : array_like, optional
            Initial guess for the parameters (length N). If None, then the
            initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            ValueError is raised).
        params_bounds : 2-tuple of array_like or `Bounds`, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            There are two ways to specify the bounds:
                - Instance of `Bounds` class.
                - 2-tuple of array_like: Each element of the tuple must be either
                  an array with the length equal to the number of parameters, or a
                  scalar (in which case the bound is taken to be the same for all
                  parameters). Use ``np.inf`` with an appropriate sign to disable
                  bounds on all or some parameters.
        fit_weights : None or scalar or M-length sequence or MxM array, optional
            Determines the uncertainty in ydata. If we define residuals as 
            r = ydata - f(xdata, *popt), then the interpretation of fit_weights depends on 
            its number of dimensions:
            A scalar or 1-D fit_weights should contain values of standard deviations of 
            errors in ydata. In this case, the optimized function is 
            chisq = sum((r / fit_weights) ** 2). 
            A 2-D fit_weights should contain the covariance matrix of errors in ydata. 
            In this case, the optimized function is 
            chisq = r.T @ inv(fit_weights) @ r.
            The default is None. This is equivalent of 1-D fit_weights filled with ones.
        absolute_weights : bool, optional
            If True, fit_weights is used in an absolute sense and the estimated parameter 
            covariance pcov reflects these absolute values.
            If False, only the relative magnitudes of the fit_weights values matter. 
            The default is False. 
        parabolic_dispersion : bool, optional
            Fit parabolic Kane model of band dispersion. The default is False. 
            Order: parabolic_dispersion > hyperbolic_dispersion_positive > hyperbolic_dispersion_negative
        hyperbolic_dispersion_positive : bool, optional
            Fit hyperbolic model (upward hyperbola) of band dispersion. The default is False.
        hyperbolic_dispersion_negative : bool, optional
            Fit hyperbolic model (downward hyperbola) of band dispersion. The default is False.
        params_name : list of str, optional
            The name of the parameters. These will be used just to creat text.
            The defult is ['alpha', 'kshift', 'cbm', 'gamma'].
            For parabolic dispersion only first three will be used.

        Returns
        -------
        m_star : float
            Calculated effective mass in m_0 unit.
        popt : array
            Optimal values for the parameters so that the sum of the squared
            residuals of ``f(xdata, *popt) - ydata`` is minimized.
        pcov : 2-D array
            The estimated approximate covariance of popt. 

        Raises
        ------
        ValueError
            if none of *_dispersion_* option is supplied.
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.
        RuntimeError
            if the least-squares minimization fails.
        OptimizeWarning
            if covariance of the parameters can not be estimated.
             
        """
        if (parabolic_dispersion or hyperbolic_dispersion_positive or 
            hyperbolic_dispersion_negative):
            pass
        else:
            raise ValueError('No dispersion option is supplied for fitting.')
            
        _EffectiveMass.__init__(self, print_log=self.print_log_info)
        text_params_ = ','.join(params_name)

        ## parameters: ['alpha', 'kshift', 'cbm', 'gamma']
        if initial_guess_params is None: 
            middle_pos = len(band_energy)//2
            guess_alpha = (band_energy[middle_pos]-band_energy[0])/(kpath[middle_pos]-kpath[0])**2
            initial_guess_params = np.array([guess_alpha, kpath[0], 0, 0])

        if len(np.shape(params_bounds)) == 1:
            # Extra grace to add to define the upper bound of the parameters.
            # The values are choosen here worked well for some of the test cases.
            extra_upper_bound = [100, 3*abs(kpath[1]-kpath[0]), 1e-4, 10]
            params_bounds = ([0, initial_guess_params[1]-1e-2, 0, 0], 
                             [gg+extra_upper_bound[iii] for iii, gg in enumerate(initial_guess_params)])

        if parabolic_dispersion:
            text_params_ = ','.join(params_name[:3])
            initial_guess_params = initial_guess_params[:3]
            params_bounds = tuple(xx[:3] for xx in params_bounds)  

        if self.print_log_info is not None:
            print('-- Effective mass calculator:')
            if parabolic_dispersion:
                print('--- Parabolic band dispersion')
            elif hyperbolic_dispersion_positive:
                print('--- Hyperbolic band dispersion (upward curvature)')
            else:
                print('--- Hyperbolic band dispersion (downward curvature)')
            
            if self.print_log_info == 'high':
                print(f'--- Initial guesses for the parameters: {initial_guess_params}')
                print(f'--- Upper and lower bounds for the parameters: {params_bounds}')
            
        m_star, popt, pcov = self._effective_mass_calculator(kpath, band_energy, 
                                                             p0=initial_guess_params, bounds=params_bounds,
                                                             sigma=fit_weights, absolute_sigma=absolute_weights,
                                                             fit_parabola=parabolic_dispersion,
                                                             fit_hyperbola_positive=hyperbolic_dispersion_positive,
                                                             fit_hyperbola_negative=hyperbolic_dispersion_negative)
        if self.print_log_info is not None:
            if self.print_log_info in ['medium', 'high']:
                print(f'--- Optimized parameters {text_params_}: {popt}')
                print(f'--- One standard deviation errors on the parameters: {np.sqrt(np.diag(pcov))}')
            print(f'--- Effective mass = {m_star:.4f} m_0')
        return m_star, popt, pcov
    
    def fit_functions(self, kpath, optimized_parameters, 
                      parabolic_dispersion:bool=False, 
                      hyperbolic_dispersion_positive:bool=False, 
                      hyperbolic_dispersion_negative:bool=False):
        """
        Fit band dispersion functions using the optimized parameters.
        
        Parameters
        ----------
        kpath : numpy array
            k on path (A^-1) of unfolded effective band structure/band centers that will
            be fitted for calculating effective mass.
        popt : array
            Optimized values for the parameters.
        parabolic_dispersion : bool, optional
            Fit parabolic Kane model of band dispersion. The default is False. 
            Order: parabolic_dispersion > hyperbolic_dispersion_positive > hyperbolic_dispersion_negative
        hyperbolic_dispersion_positive : bool, optional
            Fit hyperbolic model (upward hyperbola) of band dispersion. The default is False.
        hyperbolic_dispersion_negative : bool, optional
            Fit hyperbolic model (downward hyperbola) of band dispersion. The default is False.
            
        Returns
        -------
        array
            Band energies for the k-path and optimized parameters.

        Raises
        ------
        ValueError
            if none of *_dispersion_* option is supplied.
             
        """
        if self.print_log_info is not None:
            print('-- Return band dispersion type in fit functions:')
            if parabolic_dispersion:
                print('--- Parabolic band dispersion')
            elif hyperbolic_dispersion_positive:
                print('--- Hyperbolic band dispersion (upward curvature)')
            else:
                print('--- Hyperbolic band dispersion (downward curvature)')

        if parabolic_dispersion:
            return _EffectiveMass._fit_parabola(kpath, *optimized_parameters)
        elif hyperbolic_dispersion_positive:
             return _EffectiveMass._fit_hyperbola_positive(kpath, *optimized_parameters)
        elif hyperbolic_dispersion_negative:
            return _EffectiveMass._fit_hyperbola_negative(kpath, *optimized_parameters)
        else:
            raise ValueError('No dispersion option is supplied for fitting.')
                
class SaveBandStructuredata:
    """
    Save band structure data class.
    """
    
    @classmethod
    def save_unfolded_pc_kpts(cls, unfolded_kpts_dat, save_dir='.', file_name='kpoints_unfolded', 
                              file_name_suffix='', print_information='low'):
        """
        Save unfolded PC-kpoints.

        Parameters
        ----------
        unfolded_kpts_dat : numpy array
            Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]
        save_dir : str or path, optional
            Directory path where to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The defult is 'kpoints_unfolded'.
        file_name_suffix : str, optional
            Suffix to add to the file name. The default is ''.
        print_information : [None,'low','medium','high'], optional
                Level of printing information. 
                The default is 'low'. If None, nothing is printed.

        Returns
        -------
        None.

        """
        _GeneralFnsDefs._save_Post_unfolded_PBZ_kpts(unfolded_kpts_dat, save_dir, file_name, 
                                                     file_name_suffix, 
                                                     print_information=print_information)
        return
    
    @classmethod
    def save_unfolded_bandstucture(cls, unfolded_bandstructure, save_dir='.', file_name='bandstructure_unfolded', 
                                   file_name_suffix='', print_information='low', is_spinor:bool=False):
        """
        Save unfolded effective band structure data.

        Parameters
        ----------
        unfolded_bandstructure : numpy ndarray
            Unfolded effective band structure.
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor]
        save_dir : str or path, optional
            Directory path where to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The defult is 'bandstructure_unfolded'.
        file_name_suffix : str, optional
            Suffix to add to the file name. The default is ''.
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
        _GeneralFnsDefs._save_Post_unfolded_bandstucture(unfolded_bandstructure, save_dir, 
                                                         file_name, file_name_suffix, 
                                                         print_information=print_information, 
                                                         is_spinor=is_spinor)
        return
      
    @classmethod
    def save_unfolded_bandcenter(cls, unfolded_bandcenter, save_dir='.', file_name='unfolded_bandcenters', 
                                 file_name_suffix='', print_information='low'):
        """
        Save band centers data.

        Parameters
        ----------
        unfolded_bandceneter : numpy ndarray
            Band cenetrs data.
            Format: [kpoint coordinate, Band center, Band width, Sum of dN]
        save_dir : str or path, optional
            Directory path where to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The defult is 'bandstructure_unfolded'.
        file_name_suffix : str, optional
            Suffix to add to the file name. The default is ''.
        print_information : [None,'low','medium','high'], optional
                Level of printing information. 
                The default is 'low'. If None, nothing is printed.

        Returns
        -------
        None.

        """
        save_data = {'save2file': False, 'fdir': save_dir, 'fname': file_name, 'fname_suffix': file_name_suffix}
        _GeneralFunctionsDefs._save_band_centers(data2save=unfolded_bandcenter, 
                                                 print_log=print_information,
                                                 save_data_f_prop=save_data)
        return
    

class Plotting(_EBSplot):
    
    def __init__(self, save_figure_dir='.'):
        """
        Intializing BandUPpy Plotting class.

        Parameters
        ----------
        save_figure_dir : str, optional
            Directory where to save the figure. The default is current directory.

        """
        self.save_figure_directory = save_figure_dir
    
    def plot_ebs(self, kpath_in_angs, unfolded_bandstructure, 
                 fig=None, ax=None, save_file_name=None, CountFig=None, 
                 Ef=None, Emin=None, Emax=None, pad_energy_scale:float=0.5, 
                 threshold_weight:float=None, mode:str="fatband", 
                 yaxis_label:str='E (eV)', special_kpoints:dict=None, plotSC:bool=True,  
                 marker='o', fatfactor=20, nE:int=100, smear:float=0.05, 
                 color='gray', color_map='viridis', show_legend:bool=True,
                 plot_colormap_bandcenter:bool=True, show_colorbar:bool=False,
                 colorbar_label:str=None, vmin=None, vmax=None, 
                 show_plot:bool=True,**kwargs_savefig):
        """
        Scatter/density/band_centers plot of the band structure.

        Parameters
        ----------
        kpath_in_angs : array
            k on path (in A^-1) coordinate.
        unfolded_bandstructure : ndarray
            Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor] or
            Format: [kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        fig : matplotlib.pyplot figure instance, optional
            Figure instance to plot on. The default is None.
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on. If None, new figure will be created.
            The default is None.
        save_file_name : str, optional
            Name of the figure file. If None, figure will be not saved. 
            The default is None.
        CountFig: int, optional
            Figure count. The default is None.
        Ef : float, optional
            Fermi energy. If None, set to 0.0. The default is None.
        Emin : float, optional
            Minimum in energy. The default is None.
        Emax : float, optional
            Maximum in energy. The default is None.
        pad_energy_scale: float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        threshold_weight : float, optional
            The band centers with band weights lower than the threshhold weights 
            are discarded. The default is None. If None, this is ignored.
        mode : ['fatband','density', 'band_centers'], optional
            Mode of plot. The default is "fatband".
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plotSC : bool, optional
            Plot supercell bandstructure. The default is True.
        marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style. Marker can be either an instance of the class or 
            the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color of plot of unfolded band structure. The color of supercell
            band structures is gray. The default is 'gray'.
        color_map: str/ matplotlib colormap
            Colormap for density plot. The default is viridis.
        show_legend : bool
            If show legend or not. The default is True.
        plot_colormap_bandcenter : bool, optional
            If plotting the band ceneters by colormap. The default is True.
        show_colorbar : bool, optional
            Plot the colorbar in the figure or not. If fig=None, this is ignored.
            The default is False.
        colorbar_label : str, optional
            Colorbar label. The default is None. If None, ignored.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers. 
            By default, the colormap covers the complete value range of the supplied data.
        show_plot : bool, optional
            To show the plot when not saved. The default is True.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure instance. If ax is not None previously generated/passed fig instance
            will be returned. Return None, if no fig instance is inputed along with ax.
        ax : Axis instance
            Figure axis instance.
        CountFig: int or None
            Figure count.

        """
        print('- Plotting band structures...')
        _EBSplot.__init__(self, kpath_in_angs=kpath_in_angs, 
                         unfolded_bandstructure=unfolded_bandstructure, 
                         save_figure_dir=self.save_figure_directory)

        return self._plot(fig=fig, ax=ax, save_file_name=save_file_name, CountFig=CountFig, Ef=Ef, 
                          Emin=Emin, Emax=Emax, pad_energy_scale=pad_energy_scale, 
                          threshold_weight=threshold_weight, mode=mode,
                          yaxis_label=yaxis_label, special_kpoints=special_kpoints, 
                          plotSC=plotSC, marker=marker, fatfactor=fatfactor, nE=nE, 
                          smear=smear, color=color, color_map=color_map,
                          plot_colormap_bandcenter=plot_colormap_bandcenter,
                          show_legend=show_legend, show_colorbar=show_colorbar,
                          colorbar_label=colorbar_label, vmin=vmin, vmax=vmax, 
                          show_plot=show_plot, **kwargs_savefig)
    
    def plot_scf(self, kpath_in_angs, unfolded_bandstructure, al_scf_data, 
                 plot_max_scf_steps:int=None, save_file_name=None, 
                 Ef=None, Emin=None, Emax=None, 
                 pad_energy_scale:float=0.5, threshold_weight:float=None, 
                 yaxis_label:str='E (eV)', special_kpoints:dict=None, 
                 plot_sc_unfold:bool=True, marker='o', fatfactor=20, 
                 smear:float=0.05, color='gray', color_map='viridis', 
                 show_legend:bool=True, plot_colormap_bandcenter:bool=True, 
                 show_colorbar:bool=True, colorbar_label:str=None, 
                 vmin=None, vmax=None, show_plot:bool=True, **kwargs_savefig):
        """
        Band centers all scf steps plot.

        Parameters
        ----------
        kpath_in_angs : array
            k on path (in A^-1) coordinate. 
        unfolded_bandstructure : ndarray
            Unfolded effective band structure data. 
            Format: [k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor]
        al_scf_data : dictionary
            All SCF data.
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_data_scf is false.
            Format: {kpoint_index: {SCF_cycle_index: [Band center, Band width, Sum of dN]}}
        plot_max_scf_steps : int, optional
            How many maximum scf cycle to plot?
            The default is maximum SCF steps found in the dictionary of all k-points.
            If scf cycle not found for a particular kpoint previous SCF cycle will be plotted.
        save_file_name : str, optional
            Name of the figure file. If None, figure will be not saved. 
            The default is None.
        Ef : float, optional
            Fermi energy. If None, set to 0.0. The default is None.
        Emin : float, optional
            Minimum in energy. The default is None.
        Emax : float, optional
            Maximum in energy. The default is None.
        pad_energy_scale: float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        threshold_weight : float, optional
            The band centers with band weights lower than the threshhold weights 
            are discarded. The default is None. If None, this is ignored.
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plot_sc_unfold : bool, optional
            Plot supercell unfolded bandstructure. The default is True.
        marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style. Marker can be either an instance of the class or 
            the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color for band centers plot when color_map is not used. 
            The default is 'gray'. The color of supercell
            band structures is gray always.
        color_map: str/ matplotlib colormap
            Colormap for band centers plot. The default is viridis.
        plot_colormap_bandcenter : bool, optional
            If plotting the band ceneters by colormap. The default is True.
        show_legend : bool, optional
            If show legend or not. The default is True.
        show_colorbar : bool, optional
            Plot the colorbar in the figure or not. If fig=None, this is ignored.
            The default is True.
        colorbar_label : str, optional
            Colorbar label. The default is None. If None, ignored.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers. 
            By default, the colormap covers the complete value range of the supplied data.
        show_plot : bool, optional
            To show the plot when not saved. The default is True.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.
        
        Raises
        ------
        ValueError
            If plot mode is unknown.

        """
        print('- Plotting band centers in band structures...')
        _EBSplot.__init__(self, kpath_in_angs=kpath_in_angs, 
                         unfolded_bandstructure=unfolded_bandstructure, 
                         save_figure_dir=self.save_figure_directory)
        
        return self._plot_scf(al_scf_data, plot_max_scf_steps=plot_max_scf_steps, 
                             save_file_name=save_file_name, Ef=Ef, Emin=Emin, 
                             Emax=Emax, pad_energy_scale=pad_energy_scale, 
                             threshold_weight=threshold_weight, 
                             yaxis_label=yaxis_label, special_kpoints=special_kpoints,
                             plot_sc_unfold=plot_sc_unfold, marker=marker, 
                             fatfactor=fatfactor, smear=smear, color=color, 
                             color_map=color_map, plot_colormap_bandcenter=plot_colormap_bandcenter,
                             show_legend=show_legend, show_colorbar=show_colorbar,
                             colorbar_label=colorbar_label, vmin=vmin, vmax=vmax, 
                             show_plot=show_plot, **kwargs_savefig)
        

