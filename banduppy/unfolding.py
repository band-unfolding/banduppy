import numpy as np
from .src import BandFolding, BandUnfolding
from .Utilities import EBSplot 

### ===========================================================================    
class Unfolding(BandFolding, BandUnfolding, EBSplot):
    """
    Band folding from primitive to supercell.

    """
    def __init__(self, supercell=None, print_log='low'):
        """
        Initialize the Folding class.

        Parameters
        ----------
        supercell : 3X3 matrix, optional
            Primitive-to-supercell transformation matrix. The default is Identity matrix.
        print_log : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """       
        super().__init__(supercell=supercell, print_info=print_log)
        
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
        return self.generate_SC_K_from_pc_k_path(pathPBZ=pathPBZ, nk=nk, labels=labels, 
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
        return self.generate_K_from_k(kpointsPBZ=kpointsPBZ, kpts_weights=kpts_weights,
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
            save2file :: Save unfolded kpoints or not? 
            fir :: str or path
                Directory path where to save the file.
            fname :: str
                Name of the file.
            fname_suffix :: str
                Suffix to add to the file name.
            The default is {'save2file': False, 'fdir': '.', 'fname': 'kpoints_unfolded', 'fname_suffix': ''}.
        save_unfolded_bandstr : dictionary, optional
            save2file :: Save unfolded kpoints or not? 
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
            Format: k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
        numpy ndarray
            Unfolded effective band structure k-path.
            Format: k on path (A^-1)

        """
        if (PBZ_kpts_list_full is None) or \
            (SBZ_kpts_list in None) or (SBZ_PBZ_kpts_map is None):
            BandUnfolding.__init__(self, self.transformation_matrix, 
                                   self.PBZ_kpts_list_org, self.SBZ_kpts_list, 
                                   self.SBZ_PBZ_kpts_mapping,
                                   print_info=self.print_information)
        else:
            BandUnfolding.__init__(self, self.transformation_matrix, 
                                   PBZ_kpts_list_full, SBZ_kpts_list, 
                                   SBZ_PBZ_kpts_map, 
                                   print_info=self.print_information)
        
        return self.unfold(bandstructure, 
                           kline_discontinuity_threshold = kline_discontinuity_threshold,
                           save_unfolded_kpts = save_unfolded_kpts,
                           save_unfolded_bandstr = save_unfolded_bandstr)
    
    def plot_ebs(self, kpath_in_angs=None, unfolded_bandstructure=None, save_figure_dir='.',
                 save_file_name=None, CountFig=None, Ef=None, Emin=None, Emax=None, 
                 pad_energy_scale:float=0.5, mode:str="fatband", yaxis_label:str='E (eV)', 
                 special_kpoints:dict=None, plotSC:bool=True, fatfactor=20, nE:int=100, 
                 smear:float=0.05, scatter_color='gray', color_map='viridis'):
        """
        Scatter/density plot of the band structure.

        Parameters
        ----------
        kpath_in_angs : array, optional
            k on path (in A^-1) coordinate. The default is None.
        unfolded_bandstructure : ndarray, optional
            Unfolded effective band structure data. 
            Format: k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
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
        mode : ['fatband','density'], optional
            Mode of plot. The default is "fatband".
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plotSC : bool, optional
            Plot supercell bandstructure. The default is True.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        scatter_color : str/color, optional
            Color of scatter plot of unfolded band structure. The color of supercell
            band structures is gray. The default is 'gray'.
        color_map: str/ matplotlib colormap
            Colormap for density plot. The default is viridis.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure instance.
        ax : Axis instance
            Figure axis instance.
        CountFig: int or None
            Figure count.

        """
        
        EBSplot.__init__(self, kpath_in_angs=kpath_in_angs, 
                         unfolded_bandstructure=unfolded_bandstructure, 
                         save_figure_dir=save_figure_dir)
        
        fig, ax, CountFig = \
        self.plot(save_file_name=save_file_name, CountFig=CountFig, Ef=Ef, 
                  Emin=Emin, Emax=Emax, pad_energy_scale=pad_energy_scale, mode=mode,
                  yaxis_label=yaxis_label, special_kpoints=special_kpoints, 
                  plotSC=plotSC, fatfactor=fatfactor, nE=nE, smear=smear, 
                  scatter_color=scatter_color, 
                  color_map=color_map)
        return fig, ax, CountFig

