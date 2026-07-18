import numpy as np
import matplotlib.pyplot as plt
from .src import _BandFolding, _BandUnfolding, _GeneralFnsDefs
from .Utilities import _GeneralFunctionsDefs, _EBSplot, _FoldingDegreePlot
from .Utilities import _BandCentersBroadening, _EffectiveMass, _alloy_scattering_params

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
        if isinstance(print_log, str): print_log = print_log.lower()
        _BandFolding.__init__(self, supercell=supercell, print_info=print_log)
        
    def propose_maximum_minimum_folding(self, pathPBZ, min_num_pts:int=5, max_num_pts:int=20,
                                        serach_mode:str='brute_force', draw_plots:bool=True,  
                                        left_yaxis_label:str='Number of SC Kpoints',
                                        right_yaxis_label:str='Folding degree (%)',
                                        xaxis_label:str='Number of PC kpoints', 
                                        CountFig=None, line_color=('k', 'r'),
                                        show_plot:bool=True, save_dir='.', save_file_name=None):
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
        left_yaxis_label : str, optional
            Left yaxis label. The default is 'Number of SC Kpoints'. 
        right_yaxis_label : str, optional
            Right twin yaxis label. The default is 'Folding degree (%)'. 
        xaxis_label : str, optional
            xaxis label. The default is 'Number of PC kpoints'.
        CountFig : int, optional
            Figure count. The default is None. If None, nothing is is done. Else,
            returns CountFig increased by 1.
        line_color : tuple, matplotlib color, optional
            Line color for plot on (left yaxis, right yaxis). The default is ('k', 'r').
        show_plot : bool, optional
            Whether to show the plot when save_file_name=None
        save_dir : str/path, optional
            Directory where to save the plots. The default is '.'.
        save_file_name : str, optional
            Name of the file to be saved. If None, figure is not saved. 
            The default is None. 

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
                                                   left_yaxis_label=left_yaxis_label,
                                                   right_yaxis_label=right_yaxis_label,
                                                   xaxis_label=xaxis_label, 
                                                   line_color=line_color,
                                                   show_plot=show_plot)
            return proposed_folding_results, fig, ax, CountFig
        return proposed_folding_results
        
    def generate_SC_Kpts_from_pc_k_path(self, pathPBZ=None, nk=11, labels=None, kpts_weights=None, 
                                        save_kpts:bool=False, save_dir='.', file_name:str='', 
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
        save_kpts : bool, optional
            Save the PC kpoints, SC kpoints, SC-PC kpoints mapping, and 
            Special kpoints. The default is False. 
        save_dir : str/path_object, optional
            Directory to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The default is ''.
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>
            If file_format is qe, file_name=K_POINTS_<file_name_suffix>.dat
        file_name_suffix : str, optional
            Suffix to add after the file_name. The default is ''.
        file_format : ['vasp', 'qe'], optional
            Format of the file. The default is 'vasp'. 

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
                                                  save_kpts=save_kpts, 
                                                  save_dir=save_dir, file_name=file_name, 
                                                  file_name_suffix=file_name_suffix, 
                                                  file_format=file_format)
            
    def generate_SC_Kpts_from_pc_kpts(self, kpointsPBZ=None, kpts_weights=None,
                                      save_kpts:bool=False, save_dir='.', file_name:str='', 
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
        save_kpts : bool, optional
            Save the PC kpoints, SC kpoints, SC-PC kpoints mapping, and 
            Special kpoints. The default is False.
        save_dir : str/path_object, optional
            Directory to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The default is ''.
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>
            If file_format is qe, file_name=K_POINTS_<file_name_suffix>.dat
        file_name_suffix : str, optional
            Suffix to add after the file_name. The default is ''.
        file_format : ['vasp','qe'], optional
            Format of the file. The default is 'vasp'. 
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
                                       save_kpts=save_kpts, save_dir=save_dir, 
                                       file_name=file_name, file_name_suffix=file_name_suffix, 
                                       file_format=file_format, footer_msg=footer_msg,
                                       special_kpoints_pos_labels=special_kpoints_pos_labels)
    
    def Unfold(self, bandstructure=None, 
               PBZ_kpts_list_full = None, 
               SBZ_kpts_list = None, 
               SBZ_PBZ_kpts_map = None,
               kline_discontinuity_threshold = 0.1,
               save_unfolded_kpts:dict|None = None,
               save_unfolded_bandstr:dict|None = None,
               ab_initio_code:str = 'vasp', 
               only_unfold_for_kpts_idxs:np.ndarray|list[int]|None = None,
               only_unfold_band_idx:tuple[int|None, int|None]|list[int | 
               None, int|None] = (None, None), 
               unfold_kpts_in_batch:bool = False, 
               kpt_batch_size:int = 10,
               is_wf_file_contain_only_bandstructure_section:bool=False,
               zero_weight_kp:bool = False, 
               fermi_energy:float|None = None,
               vasp_keywards:dict|None = None,
               qe_keywards:dict|None = None,
               abinit_keywards:dict|None = None, 
               gpaw_keywards:dict|None = None, 
               wannier90_keywards:dict|None = None,
               **other_ab_initio_code_related_kwargs):
        """
        Unfold the band structure.

        Parameters
        ----------
        bandstructure : BandStructure instance from irrep.bandstructure or None, optional
            BandStructure instance from irrep.bandstructure class or from any other 
            external package. The default is None.
            If None, the `bandStructure` instance is generated within banduppy according 
            to ab initio set up provided by other keywords in the following. In this cases, 
            banduppy internally calls bandstructure functionality from irrep.bandstructure 
            and add some processing on it. 
            NOTE: Passing not None 'bandstructure' ignores bandStructure instance
            generations by banduppy's internal routine. Not None 'bandstructure' 
            has higher priority.
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
        save_unfolded_kpts : dictionary or None, optional
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
            'save2file' : bool, optional
                Save unfolded kpoints data to file or not? The default is False.
            'fdir ': str or path, optional
                Directory path where to save the file. The default is current directory.
            'fname' : str, optional
                Where to save. File name (without extension). The default is 'kpoints_unfolded'.
            'fname_suffix' : str, optional
                Suffix to add to the file name. The default is no suffix.
            }
            The default is None. If None, dictionary values will be set to default.
        save_unfolded_bandstr : dictionary, optional
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
            'save2file' : bool, optional
                Save unfolded kpoints data to file or not? The default is False.
            'fdir' : str or path, optional
                Directory path where to save the file. The default is current directory.
            'fname' : str, optional
                Where to save. File name (without extension). The default is 'bandstructure_unfolded'.
            'fname_suffix' : str, optional
                Suffix to add to the file name. The default is no suffix.
            }
            The default is None. If None, dictionary values will be set to default.            
        ab_initio_code : str, optional ['vasp', 'qe', 'abinit', 'gpaw', 'wannier90']
            Ab-initio code used to generate the wavefunction file. The default is 'vasp'.
        only_unfold_for_kpts_idxs : np.ndarray|list[int]|None, optional
            List of indices of k-points to be considered (starts from 0).
            If None, all k-points will be considered. Indices outside the range 
            [0, NK) will be ignored, where NK is the total number of K-points.
             The default is None.
        only_unfold_band_idx : tuple[int|None, int|None]|list[int | None, int|None], optional
            First number is the first band to be considered (starting from 0 for the lowest band).
            If negative, it will be counted from the top, i.e., -1 for the highest band.
            Second number is the last band to be considered (Not included) (NBin for the highest band, 
            NBin-2 to exclude the two highest bands, where NBin is the total number of bands
            used in the calculation). If negative, it will be counted from the top, 
            i.e., -3 is equaivalent to NBin-3 (exclude 3 upper bands).
            The default is (None, None).
        unfold_kpts_in_batch: bool, optional
            Whether to unfold complete wave function file in sigle shot. If False,
            unfolding is performed in K-point batch. Chunk of K-point info is loaded in RAM, 
            unfolded, and then removed from RAM once unfolding is done. This allows 
            us to aviod large memory requirement when the wave function file is 
            very large (~few GB). The default is False.
        kpt_batch_size : int (1 <= kpt_batch_size <= # of kpoints), optional
            Batch size of K-points to be unfolded in single instance. Only needed
            when unfold_kpts_in_batch = True. Smaller the batch size lower the 
            RAM requirement. Will be re-set to max and min value allowed if outside 
            range. The default is 10.    
        is_wf_file_contain_only_bandstructure_section : bool, optional
            Whether the wave function has only part of the complete bandstuture.
            Controls printing warning messages.
            This keyward is useful when user generates wavefunction file in sections due to
            memory restriction. Each section wave function file can be unfolded,
            saved unfolding data and after that remove wave function file to save
            space. Later user can merge multiple of the unfolded files to get the
            full unfolded bandstructure informations.
        zero_weight_kp : bool, optional
            Whether the K-points in ab-initio calculation corresponds to zero-weight 
            k-point method. This automatically exclude unfolding of non-zero weight
            K-points. The default is False.
        fermi_energy : float|None, optional
            User supplied Fermi-energy. If None, by default it is extracted from
            output files corresponds to specific ab-inito codes. The default is None.
        vasp_keywards : dict | None, optional
            The keywards specific to VASP ab-initio code. Will be ignored when ab_init_code != vasp.
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
             'poscar_file_path': str or file Path object, optional
                 File path containing the crystal structure in VASP (POSCAR format).
                 The default is './POSCAR'. 
             'wavecar_file_path': str or file Path object, optional
                 File path containing wave-functions in VASP (WAVECAR format).
                 The default is './WAVECAR'. 
             'vasprunxml_file_path': str or file Path object, optional
                 File path of VASP vasprun.xml file. The default is './vasprun.xml'.
             'is_spin_nondegenrate': bool, optional
                 Whether wave functions are spinors. False if they are scalars. 
                 The default is False.
             'unfold_spin_channel': str|None, optional ['up', 'dw']
                 In case of spin non degenracy which spin-channel to unfold. 
                 'up' for spin-up, 'dw' for spin-down. The default is None.
                 Must be one of the 'up' or 'dw', when is_spin_nondegenrate=True.
             'wf_cutoff_energy': float|None, optional (unit: eV)
                 Plane wave cutoff energy in eV. Not mandatory. This tag is usefull when 
                 getting plane wave related runtime error during unfolding (see FAQ).
                 The default is None.
            }
            The default is None. If None, dictionary values will be set to default.  
        qe_keywards : dict | None, optional
            The keywards specific to Quantum ESPRESSO ab-initio code. Will be ignored when ab_init_code != qe.
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
             'output_file_dir' : str or Path object, optional
                 Directory path where Quantum ESPRESSO output folder 'prefix.save' resides.
                 The default is current directory, './'. 
             'save_file_prefix' : str, optional
                 Prefix of the Quantum ESPRESSO output files (e.g. 'prefix' for 'prefix.save').
                 The default is 'prefix'.
             'unfold_spin_channel': str|None, optional ['up', 'dw']
                 In case of spin non degenracy which spin-channel to unfold. 
                 'up' for spin-up, 'dw' for spin-down. The default is None.
                 Must be one of the 'up' or 'dw' for spin polarized calculations.
             'wf_cutoff_energy': float|None, optional (unit: eV)
                 Plane wave cutoff energy in eV. Not mandatory. This tag is usefull when 
                 getting plane wave related runtime error during unfolding (see FAQ).
                 The default is None.
            }
            The default is None. If None, dictionary values will be set to default.
        abinit_keywards : dict | None, optional
            The keywards specific to ABINIT ab-initio code. Will be ignored when ab_init_code != abinit.
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
             'wfk_file_path': str or file Path object, optional
                 File path containing wave-functions in ABINIT (WFK format).
                 The default is './test_WFK'.
             'unfold_spin_channel': str|None, optional ['up', 'dw']
                 In case of spin non degenracy which spin-channel to unfold. 
                 'up' for spin-up, 'dw' for spin-down. The default is None.
                 Must be one of the 'up' or 'dw' for spin polarized calculations.
             'wf_cutoff_energy': float|None, optional (unit: eV)
                 Plane wave cutoff energy in eV. Not mandatory. This tag is usefull when 
                 getting plane wave related runtime error during unfolding (see FAQ).
                 The default is None.
            }
            The default is None. If None, dictionary values will be set to default.  
        gpaw_keywards : dict | None, optional
            The keywards specific to GPAW ab-initio code. Will be ignored when ab_init_code != gpaw.
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
             'gpaw_calculator_instance' : str or GPAW calculator object, Mandatory
                 GPAW calculator instance. The default is 'None'.
             'read_paw' : bool, optional
                 Whether to read PAW. The default is False.
             'is_spin_nondegenrate': bool, optional
                 Whether wave functions are spinors. False if they are scalars. 
                 The default is False.
             'unfold_spin_channel': str|None, optional ['up', 'dw']
                 In case of spin non degenracy which spin-channel to unfold. 
                 'up' for spin-up, 'dw' for spin-down. The default is None.
                 Must be one of the 'up' or 'dw', when is_spin_nondegenrate=True.
             'wf_cutoff_energy': float|None, optional (unit: eV)
                 Plane wave cutoff energy in eV. Not mandatory. This tag is usefull when 
                 getting plane wave related runtime error during unfolding (see FAQ).
                 The default is None.
            }
            The default is None. If None, dictionary values will be set to default. 
        wannier90_keywards : dict | None, optional
            The keywards specific to WANNIER90 ab-initio code. Will be ignored when ab_init_code != wannier90.
            Followings (key, value) dictionary pairs are allowed. If any dictionary key
            is not found, will be reset to default.
            {
             'output_file_dir' : str or Path object, optional
                 Directory path of WANNIER90 output folder where'seedname.win' file 
                 reside. The default is current directory, './'.  
             'seedname' : str, optional
                 Seedname (base filename) of WANNIER90 files (e.g. 'seedname' for 
                 'seedname.win'). The default is 'prefix'.
             'input_files_are_text_format' : bool, optional
                 The input files are text files or binary files.
                 The default is False == binary files.
             'is_spin_nondegenrate': bool, optional
                 Whether wave functions are spinors. False if they are scalars. 
                 The default is False.
             'unfold_spin_channel': str|None, optional ['up', 'dw']
                 In case of spin non degenracy which spin-channel to unfold. 
                 'up' for spin-up, 'dw' for spin-down. The default is None.
                 Must be one of the 'up' or 'dw', when is_spin_nondegenrate=True.
             'wf_cutoff_energy': float|None, optional (unit: eV)
                 Plane wave cutoff energy in eV. Not mandatory. This tag is usefull when 
                 getting plane wave related runtime error during unfolding (see FAQ).
                 The default is None.
            }
            The default is None. If None, dictionary values will be set to default. 
        ** other_ab_initio_code_related_kwargs :dict
            Any other keywards that external code such as irrep.bandstruture.from_*()
            accepts. This allows expert users finer control on the reading ab-inito
            wave function reading using irrep.bandstruture.from_*() for e.g.   
            
        Returns
        -------
        numpy ndarray
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor]
        numpy ndarray
            Unfolded effective band structure k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]

        """
        self.kline_discontinuity_threshold = kline_discontinuity_threshold
        self.save_unfolded_kpts = save_unfolded_kpts
        self.save_unfolded_bandstr = save_unfolded_bandstr
        # Print warning msg when WF file does not contain all K-points as in Kk map file
        self.is_wf_file_contain_only_bandstr_sec = is_wf_file_contain_only_bandstructure_section
        
        if PBZ_kpts_list_full is None: PBZ_kpts_list_full = self.PBZ_kpts_list_org
        if SBZ_kpts_list is None: SBZ_kpts_list = self.SBZ_kpts_list
        if SBZ_PBZ_kpts_map is None: SBZ_PBZ_kpts_map = self.SBZ_PBZ_kpts_mapping
        
        _BandUnfolding.__init__(self, self.transformation_matrix, 
                                PBZ_kpts_list_full, SBZ_kpts_list, 
                                SBZ_PBZ_kpts_map, print_info=
                                self.print_information)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if bandstructure is None: 
            from .Parsers import _ProcessPWs
            bds = _ProcessPWs(ab_initio_code=ab_initio_code,
                              only_unfold_band_idx=only_unfold_band_idx,
                              only_unfold_for_kpts_idxs=only_unfold_for_kpts_idxs,
                              kpts_batch_unfold=unfold_kpts_in_batch,
                              zero_weight_kp=zero_weight_kp, 
                              fermi_energy=fermi_energy,
                              vasp_kwards=vasp_keywards, 
                              qe_kwards=qe_keywards, 
                              abinit_kwards=abinit_keywards,
                              gpaw_kwards=gpaw_keywards,
                              wannier90_kwards=wannier90_keywards,
                              print_log=self.print_information)
            bandstructure = bds._generate_bandstructure_instance(**other_ab_initio_code_related_kwargs)   
            self.unfold_in_batch = bds.kpts_batch_unfold_
            self.kpt_batch_size = kpt_batch_size
            self.reading_gpaw_paw = False # Specific to GPAW code
            if bds.ab_initio_code_ == 'gpaw' and bds.gpaw_kwards_['read_paw']:
                self.reading_gpaw_paw = True
            # Do not warning msg when not all K-points in Kk map file is not unfolded.
            # User wants to unfold only a few K-points.
            if only_unfold_for_kpts_idxs:
                self.is_wf_file_contain_only_bandstr_sec = True
        # Passing bandstructure class from other packages is allowed
        return self._unfold(bandstructure)

    def plot_ebs(self, fig=None, ax=None, save_figure_dir='.', save_file_name=None,  
                 CountFig=None, Ef=None, Emin=None, Emax=None, pad_energy_scale:float=0.5, 
                 threshold_weight:float=None, mode:str="fatband", yaxis_label:str='E (eV)', 
                 special_kpoints:dict=None, plotSC:bool=True, marker='o', sc_marker='o', fatfactor=20, 
                 nE:int=100, smear:float=0.05, color='gray', sc_color='gray', color_map='viridis', 
                 show_legend:bool=True, show_colorbar:bool=False, colorbar_label:str=None, 
                 vmin=None, vmax=None, show_plot:bool=True, append_plts:bool=False,
                 savefig:bool=True, **kwargs_savefig):
        
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
            Name of the figure file (with extension). File extension determines 
            figure file type. If None, figure will be not saved. 
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
            The marker style for fatband plot. Marker can be either an instance of
            the class or the text shorthand for a particular marker.
            The default is 'o'.
        sc_marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style for supercell plots. Marker can be either an 
            instance of the class or the text shorthand for a particular marker. 
            The default is 'o'.        
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color of scatter plot of unfolded band structure. The default is 'gray'.
        sc_color : str/color, optional
            Color of plot of folded supercell band structure.The default is 'gray'.
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
        append_plts : bool, optional
            Special case, when overlay of multiple plots is needed. If True, it returns 
            the figure without closing the figure instance.The default is False. 
        savefig : bool, optional
            To save the plot. Ignored when save_file_name is None. The default is True.
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
        if mode == 'band_centers':
            raise AttributeError('Plot using band_cenetrs mode is not allowed within Unfolding() class. Use Plotting() class instead.')
        _EBSplot.__init__(self, save_figure_dir=save_figure_dir)

        return self._plot(fig=fig, ax=ax, save_file_name=save_file_name, CountFig=CountFig,  
                          Ef=Ef, Emin=Emin, Emax=Emax, pad_energy_scale=pad_energy_scale, 
                          threshold_weight=threshold_weight, mode=mode, yaxis_label=yaxis_label, 
                          special_kpoints=special_kpoints, plotSC=plotSC, marker=marker,
                          sc_marker=sc_marker, fatfactor=fatfactor, nE=nE, smear=smear, 
                          color=color, sc_color=sc_color, color_map=color_map,
                          show_legend=show_legend, show_colorbar=show_colorbar,
                          colorbar_label=colorbar_label, vmin=vmin, vmax=vmax, 
                          show_plot=show_plot, append_plts=append_plts,
                          savefig=savefig, **kwargs_savefig)
    
class Properties(_BandCentersBroadening, _EffectiveMass, _alloy_scattering_params):
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
        self.space_gap = 55 # add white space for text line formatting

    def collect_bandstr_data_only_in_energy_window(self, unfolded_bandstructure, Ef:float=None,
                                                   Emin:float=None, Emax:float=None,
                                                   pad_energy_scale:float=0.5,
                                                   is_band_center_data:bool=False,
                                                   bandstr_is_spinor:bool=False,
                                                   min_dN_screen:float=0.0,
                                                   save_data = {'save2file': False, 
                                                                'fdir': '.',
                                                                'fname': 'bandstructure_unfolded_window',
                                                                'fname_suffix': ''}):
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
        pad_energy_scale : float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        is_band_center_data : bool, optional
            Is the data for unfolded band center? The default is False. 
        bandstr_is_spinor : bool, optional
            If the bands in wave function files are non-degenerate (spinor). 
            The default is False.
        min_dN_screen : float, optional
            The bands with band weights lower than the threshhold weights 
            are discarded. The default is 0.
        save_data : dictionary, optional
            {
            'save2file' : bool, optional
                Save unfolded kpoints data to file or not? The default is False.
            'fdir' : str or path, optional
                Directory path where to save the file. The default is current directory.
            'fname' : str, optional
                Where to save. File name (without extension). The default is 'bandstructure_unfolded'.
            'fname_suffix' : str, optional
                Suffix to add to the file name. The default is no suffix.
            }
            The default is {'save2file': False, 'fdir': '.', 'fname': 'bandstructure_unfolded_window', 'fname_suffix': ''}.
            
         Returns
         -------
         Emin : float
             Minimum in energy.
         Emax : float
             Maximum in energy
         unfolded_bandstructure_window_ : ndarray
             Unfolded effective band structure/band center data within the range. 

         """
         
        Emin, Emax, unfolded_bandstructure_window_ = \
        _GeneralFunctionsDefs._get_bandstr_data_only_in_energy_kpts_window(unfolded_bandstructure, Ef=Ef, 
                                                                           Emin=Emin, Emax=Emax,  
                                                                           pad_energy_scale=pad_energy_scale,
                                                                           is_band_center_data=is_band_center_data,
                                                                           min_dN_screen=min_dN_screen)
        if is_band_center_data:
            _GeneralFunctionsDefs._save_band_centers(data2save=unfolded_bandstructure_window_, 
                                                     print_log=self.print_log_info,
                                                     save_data_f_prop=save_data)
        else:
            if save_data['save2file']:
                _GeneralFnsDefs._save_Post_unfolded_bandstucture(unfolded_bandstructure_window_, save_data['fdir'], 
                                                                 save_data['fname'], save_data['fname_suffix'], 
                                                                 print_information=self.print_log_info, 
                                                                 is_spinor=bandstr_is_spinor)
        return Emin, Emax, unfolded_bandstructure_window_
    
    def band_centers_broadening_bandstr(self, unfolded_bandstructure, 
                                        algorithm_version:str='Mondal2025',
                                        sigma:float=1,
                                        min_sum_dNs_for_a_band:float=0.1,
                                        min_dN_pre_screening:float=1e-2,
                                        threshold_dN_2b_trial_band_center:float=None,
                                        precision_scf_band_centers:float=1e-5,
                                        err_tolerance_compare_kpts_val:float=1e-8,
                                        collect_scf_data:bool=False,
                                        save_data = {'save2file': False, 
                                                     'fdir': '.',
                                                     'fname': 'bandcenters_unfolded',
                                                     'fname_suffix': ''}):
        """
        Find band centers and broadening of the unfolded band structure.
        
        The implementation is based on the SCF algorithm of automatic band center 
        determination from the following references:
        1. Medeiros et al, PRB 89, 041407(R) (2014) 
        2. Mondal et al, TBA
        
        Original implementation: 
            https://github.com/band-unfolding/bandup/utils/post_unfolding/
            locate_band_centers_and_estimate_broadening/find_band_centers_and_broadenings.py

        Parameters
        ----------
        unfolded_bandstructure : numpy array
            Unfolded effective band structure. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor]
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
            Note: In most of the cases the default value is good enough.
        min_dN_pre_screening : float, optional
            Discard the bands which has weights below min_dN_pre_screening to start with. 
            This pre-screening step helps to minimize the data that will processed
            now on. The default is 1e-2. 
            Note: For 'Mondal2025' algorithm_version this value is not a critical parameter. 
            Adjust the default value only for fine tuning when needed.
        threshold_dN_2b_trial_band_center : float, optional
            Initial guess of the band centers based on the threshold wights. If None,
            all the band centers are used as the initial guess. The default is None. 
            Note: For 'Mondal2025' algorithm_version this is not a critical parameter. 
            Adjust the default value only for fine tuning when needed.
        precision_scf_band_centers : float, optional
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
            {
            'save2file' : bool, optional
                Save unfolded kpoints data to file or not? The default is False.
            'fdir' : str or path, optional
                Directory path where to save the file. The default is current directory.
            'fname' : str, optional
                Where to save. File name (without extension). The default is 'bandstructure_unfolded'.
            'fname_suffix' : str, optional
                Suffix to add to the file name. The default is no suffix.
            }
            The default is {'save2file': False, 'fdir': '.', 'fname': 'bandcenters_unfolded', 'fname_suffix': ''}.

        Returns
        -------
        list of array
            Each array contains the final details of band centers in a particular
            kpoint. The list contains band center details for each kpoints.
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN]
        dictionary of dictionary of array or None
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_scf_data is false.
            Format: {kpoint_index: {SCF_cycle_index: [Band center, Band width, Sum of dN]}}

        """
        _BandCentersBroadening.__init__(self, unfolded_bandstructure=unfolded_bandstructure, 
                                        algorithm_version=algorithm_version,
                                        sigma=sigma,
                                        min_sum_dNs_for_a_band=min_sum_dNs_for_a_band,
                                        min_dN_pre_screening=min_dN_pre_screening,
                                        threshold_dN_2b_trial_band_center=threshold_dN_2b_trial_band_center,
                                        precision_scf_band_centers=precision_scf_band_centers,
                                        err_tolerance_compare_kpts_val=err_tolerance_compare_kpts_val,
                                        print_log=self.print_log_info)
        return self._scfs_band_centers_broadening(collect_data_scf=collect_scf_data,
                                                  save_data=save_data)
    
    def calculate_effecfive_mass(self, kpath, band_energy, 
                                 initial_guess_params=None, 
                                 ignore_kshift_cbm_fit:bool=True,
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
        ignore_kshift_cbm_fit : bool, optional
            Ignore fitting kshift and cbm parameters during fitting. The default is True.
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

        ## parameters: ['alpha', 'kshift', 'cbm', 'gamma']
        if initial_guess_params is None: 
            middle_pos = len(band_energy)//2
            guess_alpha = (band_energy[middle_pos]-band_energy[0])/(kpath[middle_pos]-kpath[0])**2
            initial_guess_params = np.array([guess_alpha, kpath[0], 0, 0])
            
        if ignore_kshift_cbm_fit:
            params_name =  ['alpha', 'gamma']
            initial_guess_params = np.array([initial_guess_params[0], initial_guess_params[-1]])

        if len(np.shape(params_bounds)) == 1:
            # Extra grace to add to define the upper bound of the parameters.
            # The values are choosen here worked well for some of the test cases.
            if ignore_kshift_cbm_fit:
                extra_upper_bound = [100, 10]
                params_bounds = ([0, 0], 
                                 [gg+extra_upper_bound[iii] for iii, gg in enumerate(initial_guess_params)])
            else:   
                extra_upper_bound = [100, 3*abs(kpath[1]-kpath[0]), 1e-4, 10]
                params_bounds = ([0, initial_guess_params[1]-1e-2, 0, 0], 
                                 [gg+extra_upper_bound[iii] for iii, gg in enumerate(initial_guess_params)])

        if parabolic_dispersion:
            params_name = params_name[:-1]
            initial_guess_params = initial_guess_params[:-1]
            params_bounds = tuple(xx[:-1] for xx in params_bounds)

        if self.print_log_info is not None:
            log_txt_ = f'-- Effective mass calculator::\n{"--- Band dispersion":<{self.space_gap}}: '
            if parabolic_dispersion:
                log_txt_ += 'Parabolic'
            elif hyperbolic_dispersion_positive:
                log_txt_ += 'Hyperbolic (upward curvature)'
            else:
                log_txt_ += 'Hyperbolic (downward curvature)'
            print(log_txt_)
            
            if self.print_log_info == 'high':
                print(f'{"--- Parameters":<{self.space_gap}}: {params_name}')
                print(f'{"--- Initial guesses for the parameters":<{self.space_gap}}: {initial_guess_params}')
                print(f'{"--- Upper and lower bounds for the parameters":<{self.space_gap}}: {params_bounds}')
            
        m_star, popt, pcov, params_errors = self._effective_mass_calculator(kpath, band_energy, 
                                                                            ignore_kshift_cbm_fit=ignore_kshift_cbm_fit,
                                                                            p0=initial_guess_params, bounds=params_bounds,
                                                                            sigma=fit_weights, absolute_sigma=absolute_weights,
                                                                            fit_parabola=parabolic_dispersion,
                                                                            fit_hyperbola_positive=hyperbolic_dispersion_positive,
                                                                            fit_hyperbola_negative=hyperbolic_dispersion_negative)
        if self.print_log_info is not None:
            if self.print_log_info in ['medium', 'high']:
                print(f"{'--- Optimized parameters':<{self.space_gap}}: {popt}")
                print(f"{'--- One standard deviation errors on the parameters':<{self.space_gap}}: {params_errors}")
            print_text_ = f"{'--- Results':<{self.space_gap}}: effective mass (m*) = {m_star[0]:.4f} +/- {m_star[1]:.4f} m_0"
            if (not parabolic_dispersion) and (hyperbolic_dispersion_positive or hyperbolic_dispersion_negative):
                print_text_ += f'\n{" ":<{self.space_gap}}: nonparabolicity parameter (gamma) = {popt[-1]:.2f} +/- {params_errors[-1]:.2f} ev^-1'
            print(print_text_,'\n')
        return m_star, popt, pcov, params_errors
    
    def effective_mass_fit_functions(self, kpath, optimized_parameters, 
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
            log_txt_ = f'{"-- Band dispersion type in fit functions":<{self.space_gap}}: '
            if parabolic_dispersion:
                log_txt_ += 'Parabolic'
            elif hyperbolic_dispersion_positive:
                log_txt_ += 'Hyperbolic (upward curvature)'
            else:
                log_txt_ += 'Hyperbolic (downward curvature)'
            print(log_txt_, '\n')

        if parabolic_dispersion:
            if len(optimized_parameters) == 1:
                return _EffectiveMass._fit_parabola_short(kpath, *optimized_parameters)
            return _EffectiveMass._fit_parabola(kpath, *optimized_parameters)
        elif hyperbolic_dispersion_positive:
            if len(optimized_parameters) == 2:
                return _EffectiveMass._fit_hyperbola_positive_short(kpath, *optimized_parameters)
            return _EffectiveMass._fit_hyperbola_positive(kpath, *optimized_parameters)
        elif hyperbolic_dispersion_negative:
            if len(optimized_parameters) == 2:
                return _EffectiveMass._fit_hyperbola_negative_short(kpath, *optimized_parameters)
            return _EffectiveMass._fit_hyperbola_negative(kpath, *optimized_parameters)
        else:
            raise ValueError('No dispersion option is supplied for fitting.')
            
    def calculate_alloy_scattering_potential(self, m_star, unit_cell_volm, composition,
                                             band_energy, band_width,
                                             intial_guess_u0=1, fitting_bounds_u0=(0,2),
                                             non_parabolocity_param_m_star=0):
        """
        This function calculates the alloy-disordered scattering lifetime
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

        Parameters
        ----------
        m_star : float 
            Carrier effective mass (in m0 unit). E.g. 0.2
        unit_cell_volm : float
            Primitive cell volume (in Angstrom^3).
        composition : float
            Alloy mole fraction (0 <= composition <= 1).
        band_energy : float array
            Band energies (eV) to fit for scattering lifetime equation.
        band_width : float array
            Band width/broadening (eV) to fit for scattering lifetime equation..
        intial_guess_u0 : float, optional
            Intial guess for fitting parameter - scattering potential, U0. The default is 1.
        fitting_bounds_u0 : tuple, optional
            Fitting bounds for fitting parameter - scattering potential, U0. The default is (0,2).
        non_parabolocity_param_m_star : float, optional
            hyperbolicity parameter / non-parabolic parameter, gamma, (eV^-1) 
            in Kane model for nonparabolic spherical bands:
            E (1 + gamma*E) = hbar^2 k^2/(2m*)
            The default is 0.

        Returns
        -------
        Tuple (float, float)
            Fitted parameter, U0 => (U0 +- U0_std_error) in eV
            One standard deviation errors on the parameters (eV): 
            U0_std_error = np.sqrt(np.diag(pcov))
        pcov : 2D array
            The estimated approximate covariance of popt.

        """
        _alloy_scattering_params.__init__(self, m_star, unit_cell_volm, composition,
                                          non_parabolocity_param=non_parabolocity_param_m_star)
        popt, pcov, perr =  self._calculate_alloy_scattering_potential(band_energy, band_width,
                                                                       intial_guess_u0=intial_guess_u0,
                                                                       fitting_bounds_u0=fitting_bounds_u0)
        
        if self.print_log_info is not None:
            print('-- Alloy scattering potential calculator::')
            if self.print_log_info in ['medium', 'high']:
                print(f"{'--- Optimized parameters':<{self.space_gap}}: {popt}")
                print(f"{'--- One standard deviation errors on the parameters':<{self.space_gap}}: {perr}")
            print_text_ = f"{'--- Results':<{self.space_gap}}: scattering potential = {popt[0]:.4f} +/- {perr[0]:.4f} eV"
            print(print_text_,'\n')
        return (popt[0], perr[0]), pcov
    
    def alloy_scattering_lifetime_function(self, energy, m_star, unit_cell_volm, composition,
                                           scattering_potential, non_parabolocity_param_m_star=0):
        """
        This function calculates the alloy-disordered scattering lifetime, 1/tau,
        using a statistically averaged disorder potential in the matrix element for
        Fermi's golden rule.         
        
        hbar/tau = 2*pi * U0^2 *x(1-x) * Omega0 * m*^(3/2)/sqrt(2)/pi^2/hbar^3 * 
                (1+2*gamma*E) * sqrt(E(1+gamma*E)) * (1+2*gamma*E+(4/3)*gamma^2*E^2) / (1+2*gamma*E)^2
                
        Reference: 
            1. Pant et. al., APL, 117, 242105 (2020)
            2. Mondal et al, TBA 

        Parameters
        ----------            
        energy : float array
            Band energies (eV).
        m_star : float 
            Carrier effective mass (in m0 unit). E.g. 0.2
        unit_cell_volm : float
            Primitive cell volume (in Angstrom^3).
        composition : float
            Alloy mole fraction (0 <= composition <= 1).
        scattering_potential : float
            Alloy-disordered scattering potential (eV).
        non_parabolocity_param_m_star : float, optional
            hyperbolicity parameter / non-parabolic parameter, gamma, (eV^-1) 
            in Kane model for nonparabolic spherical bands:
            E (1 + gamma*E) = hbar^2 k^2/(2m*)
            The default is 0.    

        Returns
        -------
        1D array of floats
            Alloy-disordered scattering lifetime, hbar/tau (eV).

        """
        _alloy_scattering_params.__init__(self, m_star, unit_cell_volm, composition,
                                          non_parabolocity_param=non_parabolocity_param_m_star)
        return self._Fermi_rule_fit(energy, scattering_potential)
                
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
            Name of the file (without extension). The defult is 'kpoints_unfolded'.
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
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor]
        save_dir : str or path, optional
            Directory path where to save the file. The default is current directory.
        file_name : str, optional
            Name of the file (without extension). The defult is 'bandstructure_unfolded'.
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
    
    @staticmethod
    def save_merge_unfolded_kp_bd_data(unfolded_kpoints_list, unfolded_bandstructure_list,
                                       save_data:bool=True, save_dir='.', 
                                       save_file_name=['kpoints_unfolded_merged',
                                                       'bandstructure_unfolded_merged'], 
                                       print_information='low', **kwargs):
        """
        This function merges multiple unfolded kpoints and bandstructure files.
        This is useful when separate kpoints_unfolded.dat and bandstructure_unfolded.dat
        files are generated for different band structure sections (e.g. kpoints_unfolded_LG.dat,
        kpoints_unfolded_GX.dat) etc. and needed to merge them in the end for plotting.
        
        NOTE: The files will be merged in the order they appear in the list. 
        Ensure the list is arranged in the intended merge order.

        Parameters
        ----------
        unfolded_kpoints_list : list of str or Path or ndarray
            If string: Full file path containing unfolded kpoints. File names should 
            be with extensions. E.g.,
            ['./kpoints_unfolded_LG.dat', './kpoints_unfolded_GX.dat'].
            If numpy array: Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]
        unfolded_bandstructure : list of str or Path or ndarray
            If string: Full file path containing unfolded bandstructure/bandcenters. 
            File names should be with extensions. 
            E.g., ['./bandstructure_unfolded_LG.dat', './bandstructure_unfolded_GX.dat'].
            If numpy array: Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        save_data : bool, optional
            Whether to save the merged kpoints and bandstructure to files.
        save_dir : str or path, optional
            Directory path where to save the file. The default is current directory.
        save_file_name : str, optional
            Name of the file to be saved (without extension). The defult is 
            'kpoints_unfolded_merged' for merged kpoints file and 
            'bandstructure_unfolded_merged' for merged bandstructure file. 
        print_information : [None,'low','medium','high'], optional
                Level of printing information. 
                The default is 'low'. If None, nothing is printed.

        Returns
        -------
        unfolded_kpoints_ : ndarray
            Merged unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1)].
        unfolded_bandstructure_ : ndarray
            Merged unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        """
        unfolded_kpoints_, unfolded_bandstructure_ = [], []
        
        shift_kpt_angs = 0
        #Format: [k-index, k on path (A^-1)]
        for unfolded_kpoints in unfolded_kpoints_list:
            ukpt = unfolded_kpoints if isinstance(unfolded_kpoints, np.ndarray)\
                                     else np.loadtxt(unfolded_kpoints, comments='#', usecols=(0,1))
            shift_kpt_angs = ukpt[-1,1]
            ukpt[:,1] += shift_kpt_angs
            unfolded_kpoints_.append(ukpt)
        unfolded_kpoints_ = np.concat(unfolded_kpoints_, axis=0)  
        
        #Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
        #Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        shift_kpt_angs = 0
        for unfolded_bandstructure in unfolded_bandstructure_list:
            bbds = unfolded_bandstructure if isinstance(unfolded_bandstructure, np.ndarray)\
                                           else np.loadtxt(unfolded_bandstructure, comments='#')
            shift_kpt_angs = max(bbds[:,1])
            bbds[:,1] += shift_kpt_angs
            unfolded_bandstructure_.append(bbds)
        unfolded_bandstructure_ = np.concatenate(unfolded_bandstructure_, axis=0) 
        
        if save_data:
            _GeneralFnsDefs._save_Post_unfolded_PBZ_kpts(unfolded_kpoints_, save_dir,  
                                                         save_file_name[0], '', 
                                                         print_information=print_information)
    
            _GeneralFnsDefs._save_Post_unfolded_bandstucture(unfolded_bandstructure_, save_dir, 
                                                             save_file_name[1], '', 
                                                             print_information=print_information)
        
        return unfolded_kpoints_, unfolded_bandstructure_
      
    @classmethod
    def save_unfolded_bandcenter(cls, unfolded_bandcenter, save_dir='.', file_name='bandcenters_unfolded', 
                                 file_name_suffix='', print_information='low'):
        """
        Save band centers data.

        Parameters
        ----------
        unfolded_bandceneter : numpy ndarray
            Band cenetrs data.
            Format: [k index, k on path (A^-1), Band center (eV), Band width (eV), Sum of dN]
        save_dir : str or path, optional
            Directory path where to save the file. The default is current directory.
        file_name : str, optional
            Name of the file (without extension). The defult is 'bandcenters_unfolded'.
        file_name_suffix : str, optional
            Suffix to add to the file name. The default is ''.
        print_information : [None,'low','medium','high'], optional
                Level of printing information. 
                The default is 'low'. If None, nothing is printed.

        Returns
        -------
        None.

        """
        save_data = {'save2file': True, 'fdir': save_dir, 'fname': file_name, 'fname_suffix': file_name_suffix}
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
    
    @staticmethod
    def _get_unfolded_kp_bd_data(unfolded_kpoints, unfolded_bandstructure):
        """
        Generate unfolded data from files or numpy array.

        Parameters
        ----------
        unfolded_kpoints : str or Path or ndarray
            If string: Full file path containing unfolded kpoints. File names should 
            be with extensions. E.g., './kpoints_unfolded.dat'.
            If numpy array: Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]
        unfolded_bandstructure : str or Path or ndarray
            If string: Full file path containing unfolded bandstructure/bandcenters. 
            File names should be with extensions. E.g., './bandstructure_unfolded.dat'.
            If numpy array: Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers

        Returns
        -------
        unfolded_kpoints_ : ndarray
            Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1)].
        unfolded_bandstructure_ : ndarray
            Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Format: [k-index, k on path (A^-1)]
        unfolded_kpoints_ = unfolded_kpoints[:, :2] if isinstance(unfolded_kpoints, np.ndarray) else \
            np.loadtxt(unfolded_kpoints, comments='#', usecols=(0,1))
        #Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
        #Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        unfolded_bandstructure_ = unfolded_bandstructure if isinstance(unfolded_bandstructure, np.ndarray) else \
            np.loadtxt(unfolded_bandstructure, comments='#')
        return unfolded_kpoints_, unfolded_bandstructure_
    
    def plot_ebs(self, unfolded_kpoints, unfolded_bandstructure, 
                 fig=None, ax=None, save_file_name=None, CountFig=None, 
                 Ef=None, Emin=None, Emax=None, pad_energy_scale:float=0.5, 
                 threshold_weight:float=None, mode:str="fatband", 
                 yaxis_label:str='E (eV)', special_kpoints:dict=None, plotSC:bool=True,  
                 marker='o', sc_marker='o', fatfactor=20, nE:int=100, smear:float=0.05,
                 color='gray', sc_color='gray', color_map='viridis', show_legend:bool=True,
                 plot_colormap_bandcenter:bool=True, show_colorbar:bool=False,
                 colorbar_label:str=None, vmin=None, vmax=None, 
                 show_plot:bool=True, append_plts:bool=False,
                 savefig:bool=True, **kwargs_savefig):
        """
        Scatter/density/band_centers plot of the band structure.

        Parameters
        ----------
        unfolded_kpoints : str or Path or ndarray
            If string: Full file path containing unfolded kpoints. File names should 
            be with extensions. E.g., './kpoints_unfolded.dat'.
            If numpy array: Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]
        unfolded_bandstructure : str or Path or ndarray
            If string: Full file path containing unfolded bandstructure/bandcenters. 
            File names should be with extensions. E.g., './bandstructure_unfolded.dat'.
            If numpy array: Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or
            Format: [k index, kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        fig : matplotlib.pyplot figure instance, optional
            Figure instance to plot on. The default is None.
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on. If None, new figure will be created.
            The default is None.
        save_file_name : str, optional
            Name of the figure file (with extension). File name extension determines the 
            figure file type to be saved. If None, figure will be not saved. 
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
        sc_marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style for supercell plots. Marker can be either an 
            instance of the class or the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color of plot of unfolded band structure. The default is 'gray'.
        sc_color : str/color, optional
            Color of plot of folded supercell band structure.The default is 'gray'.
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
        append_plts : bool, optional
            Special case, when overlay of multiple plots is needed. If True, it returns 
            the figure without closing the figure instance.The default is False. 
        savefig : bool, optional
            To save the plot. Ignored when save_file_name is None. The default is True.
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
        unfolded_kpts, unfolded_bandstr = self._get_unfolded_kp_bd_data(unfolded_kpoints, 
                                                                        unfolded_bandstructure)
        _EBSplot.__init__(self, unfolded_kpoints=unfolded_kpts, 
                          unfolded_bandstructure=unfolded_bandstr, 
                          save_figure_dir=self.save_figure_directory)

        return self._plot(fig=fig, ax=ax, save_file_name=save_file_name, CountFig=CountFig, Ef=Ef, 
                          Emin=Emin, Emax=Emax, pad_energy_scale=pad_energy_scale, 
                          threshold_weight=threshold_weight, mode=mode,
                          yaxis_label=yaxis_label, special_kpoints=special_kpoints, 
                          plotSC=plotSC, marker=marker, sc_marker=sc_marker, 
                          fatfactor=fatfactor, nE=nE, smear=smear, 
                          color=color, sc_color=sc_color, color_map=color_map,
                          plot_colormap_bandcenter=plot_colormap_bandcenter,
                          show_legend=show_legend, show_colorbar=show_colorbar,
                          colorbar_label=colorbar_label, vmin=vmin, vmax=vmax, 
                          show_plot=show_plot, append_plts=append_plts,
                          savefig=savefig, **kwargs_savefig)
    
    def plot_scf(self, al_scf_data, unfolded_kpoints, unfolded_bandstructure, 
                 plot_max_scf_steps:int=None, save_file_name=None, 
                 Ef=None, Emin=None, Emax=None, 
                 pad_energy_scale:float=0.5, threshold_weight:float=None, 
                 yaxis_label:str='E (eV)', special_kpoints:dict=None, 
                 plot_sc_unfold:bool=True, marker='o', fatfactor=20, 
                 smear:float=0.05, color='gray', color_map='viridis', 
                 show_legend:bool=True, plot_colormap_bandcenter:bool=True, 
                 show_colorbar:bool=True, colorbar_label:str=None, 
                 vmin=None, vmax=None, show_plot:bool=True,
                 savefig:bool=True, **kwargs_savefig):
        """
        Band centers all scf steps plot.

        Parameters
        ----------
        al_scf_data : dictionary
            All SCF data.
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_data_scf is false.
            Format: {kpoint_index: {SCF_cycle_index: [Band center, Band width, Sum of dN]}}
        unfolded_kpoints : str or Path or ndarray
            If string: Full file path containing unfolded kpoints. File names should 
            be with extensions. E.g., './kpoints_unfolded.dat'.
            If numpy array: Unfolded kpoints in k-path.
            Format: [k-index, k on path (A^-1), k1, k2, k3]
        unfolded_bandstructure : str or Path or ndarray
            If string: Full file path containing unfolded bandstructure/bandcenters. 
            File names should be with extensions. E.g., './bandstructure_unfolded.dat'.
            If numpy array: Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy (eV), weight, "Sx, Sy, Sz" if spinor] or 
        plot_max_scf_steps : int, optional
            How many maximum scf cycle to plot?
            The default is maximum SCF steps found in the dictionary of all k-points.
            If scf cycle not found for a particular kpoint previous SCF cycle will be plotted.
        save_file_name : str, optional
            Name of the figure file (with extension). File name extension determines the 
            figure file type to be saved. If None, figure will be not saved. 
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
        savefig : bool, optional
            To save the plot. Ignored when save_file_name is None. The default is True.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.
        
        Raises
        ------
        ValueError
            If plot mode is unknown.

        """
        print('- Plotting band centers in band structures...')
        unfolded_kpts, unfolded_bandstr = self._get_unfolded_kp_bd_data(unfolded_kpoints, 
                                                                        unfolded_bandstructure)
        _EBSplot.__init__(self, unfolded_kpoints=unfolded_kpts, 
                          unfolded_bandstructure=unfolded_bandstr, 
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
                             show_plot=show_plot, savefig=savefig, **kwargs_savefig)

    def save_plot_figure(self, fig_name, fig=None, savefig:bool=True, show_plot:bool=True,
                         CountFig=None, **kwargs_savefig):
        """
        Saving generated plot/figure.

        Parameters
        ----------
        fig_name : str, optional
            Name of the figure file (with extension). File name extension determines the 
            figure file type to be saved. If None, figure will be not saved.
            The default is None.
        fig : matplotlib.pyplot figure instance, optional
            Figure instance to plot on. The default is None.
        savefig : bool, optional
            To save the plot. Ignored when save_file_name is None. The default is True.
        show_plot : bool, optional
            To show the plot when not saved. The default is True.
        CountFig: int, optional
            Figure count. The default is None.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.

        Returns
        -------
        CountFig: int or None
            Figure count.
        """
        if not savefig:
            if show_plot: plt.show()
            return CountFig

        if fig is not None:
            fig.savefig(f'{self.save_figure_directory}/{fig_name}',
                        bbox_inches='tight', **kwargs_savefig)
        else:
            plt.savefig(f'{self.save_figure_directory}/{fig_name}',
                        bbox_inches='tight', **kwargs_savefig)
        if CountFig is not None: CountFig += 1
        plt.close()
        return CountFig
        

