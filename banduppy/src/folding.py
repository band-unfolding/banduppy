import numpy as np
from ..BasicFunctions.general_functions import SaveData2File, _BasicFunctionsModule, _draw_line_length
from .folding_properties import FindProperties
from .. import __version__
    
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
    
## ============================================================================    
class _KpointsModule:   
    
    @staticmethod
    def _find_K_from_k(k, transformation_matrix):
        """
        Generate list of SC K-vectors onto which the PC k-vectors folds. 
        
            K = k.P

        Parameters
        ----------
        k : ndarray of floats
            PC kpoints.
        transformation_matrix : ndarray
            PC to SC transformation matrix.

        Returns
        -------
        ndarray of floats
            SC kpoints.

        """
        return np.dot(k, transformation_matrix.T)

    @classmethod   
    def _remove_duplicate_kpoints(cls, kpoints, sort_kpts:bool=False, 
                                  return_id_mapping:bool=True, decimals:int=10):
        """
        Remove duplicate kpoints.

        Parameters
        ----------
        kpoints : ndarray of floats
            kpoints list.
        sort_kpts : bool, optional
            Sort the kpoints. The default is False.
        return_id_mapping : bool, optional
            Return the id of reverse mapping. The default is True.
        decimals : int, optional
            How many decomal points to round up. The default is 10.

        Returns
        -------
        ndarray of floats, ndarray of int
            Reduced kpoint list.

        """
        kpoints = np.asarray(kpoints)
        unique_kpts, unique_kpts_ids, kpts_ids = np.unique(_BasicFunctionsModule._return_mod(kpoints),
                                                           axis=0, return_index=True, return_inverse=True)
        
        if sort_kpts:
            sort_pos_kpts = np.argsort(unique_kpts_ids)
            unique_kpts_ids = unique_kpts_ids[sort_pos_kpts]
            unique_kpts = kpoints[unique_kpts_ids]
            kpts_ids = kpts_ids[sort_pos_kpts]

        if return_id_mapping:
            return unique_kpts, kpts_ids
        else:
            return unique_kpts
    
    @classmethod
    def _search_folding_percent_brute_force(cls, transformation_matrix, start, end, 
                                            min_div_points, max_div_points):  
        """
        Calculates SC Kpoints from PC kpoints and returns percent of folding.
        This method uses brute force to calculate folding degree i.e. complete
        list of SC Kpoints are generated in the process. 
        (Loop over nkpoints in the path segment starting with min_div_points until
        max_div_points with increment of 1.)
        
        Folding percent = ((#of PC kpoints - #of folded SC Kpoints)/(#of PC kpoints))*100

        Parameters
        ----------
        transformation_matrix : ndarray
            PC to SC transformation matrix.
        start : 1d array
            Starting coordinate of kpoints path.
        end : 1d array
            End coordinate of kpoints path.
        min_div_points : int
            Minimum number of kpoints division in the k-path.
        max_div_points : int
            Maximum number of kpoints division in the k-path.

        Returns
        -------
        numpy ndarray
            2d array with each row containing number of division in the 1st
            column and percent of folding in the 2nd column.

        """
        folding_info_list = []
        # Loop over nkpoints in the path segment starting with min_div_points until
        # max_div_points with increment of 1.
        for div_points in range(min_div_points, max_div_points): 
            len_unique_Kpts = \
            len(cls._remove_duplicate_kpoints(
                _KpointsModule._find_K_from_k(np.linspace(start, end, div_points), 
                                                            transformation_matrix),
                sort_kpts=False, return_id_mapping=False))
            
            folding_info_list.append([div_points, 
                                      (div_points-len_unique_Kpts)/div_points*100]) 
        return np.array(folding_info_list)

    @classmethod           
    def _serach_max_min_folding(cls, transformation_matrix, start, end, 
                                min_num_pts:int=5, max_num_pts:int=20,
                                serach_mode:str='brute_force'):
        """
        Calculates SC Kpoints from PC kpoints and returns percent of folding.
        Maximum and Minimum degree of folding are reported.
        
        Folding percent = ((#of PC kpoints - #of folded SC Kpoints)/(#of PC kpoints))*100

        Parameters
        ----------
        transformation_matrix : ndarray
            PC to SC transformation matrix.
        start : 1d array
            Starting coordinate of kpoints path.
        end : 1d array
            End coordinate of kpoints path.
        min_num_pts : int, optional
            Minimum number of kpoints division in the k-path. The default is 5.
        max_num_pts : int, optional
            Maximum number of kpoints division in the k-path. The default is 20.
        serach_mode : ['brute_force'], optional
            Method to calculate SC Kpoints. The default is 'brute_force'.

        Returns
        -------
        folding_data_ : numpy ndarray
            2d array with each row containing number of division in the 1st
            column and percent of folding in the 2nd column.

        """
        assert max_num_pts >= min_num_pts, f'max_num_pts should be >= min_num_pts {min_num_pts}.'
        if serach_mode == 'brute_force':
            folding_data_ = cls._search_folding_percent_brute_force(transformation_matrix, 
                                                                    start, end, 
                                                                    min_num_pts, max_num_pts)
            max_folding = folding_data_[np.argmax(folding_data_[:,1])]
            min_folding = folding_data_[np.argmin(folding_data_[:,1])]
            print(f'--- Maximum folding (nkpts, folding percent): {int(max_folding[0]):>3d}, {max_folding[1]:.3f}%')
            print(f'--- Minimum folding (nkpts, folding percent): {int(min_folding[0]):>3d}, {min_folding[1]:.3f}%')
            return folding_data_ 
        
    @classmethod           
    def _propose_max_min_folding(cls, transformation_matrix, pathPBZ, 
                                 min_num_pts:int=5, max_num_pts:int=20,
                                 serach_mode:str='brute_force'):
        """
        Calculates SC Kpoints from PC kpoints and returns percent of folding.
        Maximum and Minimum degree of folding are reported.
        
        Folding percent = ((#of PC kpoints - #of folded SC Kpoints)/(#of PC kpoints))*100

        Parameters
        ----------
        transformation_matrix : ndarray
            PC to SC transformation matrix.
        pathPBZ : ndarray/list
            PC kpoint path nodes in reduced coordinates.
        min_num_pts : int, optional
            Minimum number of kpoints division in the k-path. The default is 5.
        max_num_pts : int, optional
            Maximum number of kpoints division in the k-path. The default is 20.
        serach_mode : ['brute_force'], optional
            Method to calculate SC Kpoints. The default is 'brute_force'.

        Returns
        -------
        propose_folding_data_ : dictionary
            {index: ((start node, end node), folding data)}
            index : Index of path segment searched from the pathPBZ list supplied.
            folding data : 2d array with each row containing number of division in the 1st
            column and percent of folding in the 2nd column.

        """
        propose_folding_data_ = {}
        print(f"{'='*_draw_line_length}\n- Proposing maximum and minimum band folding...")
        # Iterate over the list, skipping the last element
        for i in range(len(pathPBZ) - 1):
            # Check if both current and next elements are not None
            if pathPBZ[i] is not None and pathPBZ[i + 1] is not None:
                # Get start and end of k-path segment
                start, end = np.array(pathPBZ[i]), np.array(pathPBZ[i + 1])
                assert start.shape == end.shape == (3,)
                print(f'-- k-path: {start} --> {end}')
                # Search for max-min folding
                propose_folding_data_[i] = ((start, end),
                    cls._serach_max_min_folding(transformation_matrix, start, end, 
                                                                       min_num_pts=min_num_pts, 
                                                                       max_num_pts=max_num_pts,
                                                                       serach_mode=serach_mode))
        return propose_folding_data_

        
    @staticmethod   
    def _generate_kpts_from_kpath(pathPBZ, nk, labels):
        """
        Creates

        Parameters
        ----------
        pathPBZ : ndarray/list, optional
            PC kpoint path nodes in reduced coordinates. 
        nk : int ot tuple, optional
            Number of kpoints in each k-path segment. 
        labels : string ot list of strings
            Labels of special k-points, either as a continuous list or string. 
            Do not use ',' or multidimentional list do define disjoint segmants.
            e.g. Do not use labels='LG,KG'. Use labels='LGKG'. The 'None' in the
            pathPBZ will take care of the disjoint segments.
            If multiple word needs to be single label, use list.
            e.g. labels=['X','Gamma','L']. Do not use string labels='XGammaL'.
            If None, the special kpoints will be indexed as 1,2,3,...

        Returns
        -------
        kpointsPBZ : ndarray, optional
            PC kpoint list. 
        special_kpoints_pos_labels : dictionary
            Special k-points position and labels.

        """
        # List of number_of_kpoints in segment
        # Convert nk to generator if it's iterable
        nkgen = (x for x in nk) if isinstance(nk, Iterable) else (nk for _ in pathPBZ)
        
        # Total nodes in the PC BZ supplied
        total_k_nodes = len([1 for k in pathPBZ if k is not None])
        
        # Create labels list for special_kpoints_labels
        if labels is None:
            labels = [str(i+1) for i in range(total_k_nodes)]
        else:
            assert len(labels) >= total_k_nodes , 'Number of labels is less than the supplied k-path segments.'
        labels = (l for l in labels)
        labels = [None if k is None else next(labels) for k in pathPBZ]
        
        # Initialize
        kpointsPBZ=np.zeros((0,3))
        special_kpoints_pos_labels={}
        
        # Iterate over the list, skipping the last element
        for i in range(len(pathPBZ) - 1):
            # Check if both current and next elements are not None
            if pathPBZ[i] is not None and pathPBZ[i + 1] is not None:
                # Get position of special k-point for plotting later
                special_kpoints_pos_labels[kpointsPBZ.shape[0]] = labels[i]
                # Get start and end of k-path segment
                start, end = np.array(pathPBZ[i]), np.array(pathPBZ[i + 1])
                assert start.shape == end.shape == (3,)
                # Get nk in k-path segment
                try:
                    num_points = next(nkgen)
                except: 
                    raise StopIteration('Number of nk is less than the supplied k-path segments.')
                # Generate k-points using linspace
                interpolated_points = np.linspace(start, end, num_points)
                kpointsPBZ = np.vstack((kpointsPBZ, interpolated_points))
                # Get position of special k-point for plotting later
                special_kpoints_pos_labels[kpointsPBZ.shape[0] - 1] = labels[i+1] 
    
        return kpointsPBZ, special_kpoints_pos_labels, nkgen, labels
    
    @staticmethod
    def _pad_kpts_weights(kpoints, kpts_weights=1):
        """
        Padding the weights of SC kpoints.

        Parameters
        ----------
        kpoints : ndarray of floats
            kpoints list.
        kpts_weights : int or float or 1d numpy array, optional
            Weights of the SC kpoints. The default is 1.

        Returns
        -------
        ndarray of floats
            kpoints list padded with weights.

        """

        if isinstance(kpts_weights, int) or isinstance(kpts_weights, float):
            append_weights = np.array([[kpts_weights]*len(kpoints)])
        else:
            append_weights = np.copy([kpts_weights])
        return np.concatenate((kpoints, append_weights.T), axis=1)
    
    @staticmethod
    def _generate_foot_text(pc_kpoints_list, labels=None, nk_list=None):
        """
        Create foorter text for the SC kpoints save file.

        Parameters
        ----------
        pc_kpoints_list : ndarray/list
            List of PC kpoints or k-path.
        labels : list, optional
            List of special k-points in k-path. The default is None.
        nk_list : list, optional
            List of number of k-points in each k-path segments. The default is None.

        Returns
        -------
        append_foot_file : str
            Text string for footer of save file.

        """
        append_foot_file  = '\n\n! The above SCKPTS (and/or some other SCKPTS related to them by symm. ops. of the SCBZ)'
        append_foot_file += '\n! unfold onto the pckpts listed below (selected by you):'
        append_foot_file += '\n! k-points for PC bandstructure '
        # Add label texts
        if labels is not None:                   
            append_foot_file += ','.join('-'.join(label for label in group_ if label) 
                                         for group_ in ''.join(str(label) for label in labels).split('None') if group_)         
        # Add number of kpoints in each segment
        if nk_list is not None: 
            if not isinstance(nk_list, Iterable): nk_list = [nk_list]
            append_foot_file += '\n! '
            append_foot_file += ' '.join(str(nk) for nk in nk_list)
            append_foot_file += "\n! reciprocal"
        # Add kpoints    
        for i, kp in enumerate(pc_kpoints_list):
            append_foot_file += '\n!'
            if kp is None:
                append_foot_file += ' '
            else:
                append_foot_file += ' '.join(str(f'{kppp:12.8f}') for kppp in kp)
                if labels is not None: append_foot_file += f'   {labels[i]}'
        return append_foot_file
    
    def _generate_header_text(self, save_all_kpts:bool=False):
        """
        Create text that will be added in the front of file.

        Parameters
        ----------
        save_all_kpts : bool, optional
            Generate header text for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints. 
            The default is False. 

        Returns
        -------
        header_msg : dictionary
            Text string for header of save file. If save_all_kpts is True,
            generate header text for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints; else generate header
            text only for SC kpoints.

        """
        # Generate header text for SC kpoints
        header_msg = {}
        header_msg['SC']  = f"K-points for SC bandstructure generated using banduppy-{__version__} package"
        header_msg['SC'] += f"\n{len(self.SBZ_kpts_list)}\nreciprocal"
        
        header_msg['SpecialKpoints']   = f"Special SC kpoints indices generated using banduppy-{__version__} package"
        header_msg['SpecialKpoints']  += "Kpoints index: Kpoints lebel"
                
        # Generate header text for PC kpoints, SC-PC kpoints mapping, and special kpoints  
        if save_all_kpts:
            # Create text that will be added in the front of file
            header_msg['PC']  = f"k-points for PC bandstructure generated using banduppy-{__version__} package"
            header_msg['PC'] += f"\n{len(self.PBZ_kpts_list_org)}\nreciprocal"
            
            header_msg['SCPC_map']  = f"Mapping for SC Kpoints to PC kpoints indices generated using banduppy-{__version__} package"
            header_msg['SCPC_map'] += "K-k relation: (K index: K -> k index unique: k unique -> k index: k)"
            
        return header_msg
    
    def _generate_footer_text(self, footer_text=None, save_all_kpts:bool=False):
        """
        Create text that will be added in the bottom of file.

        Parameters
        ----------
        footer_text: str, optional
            Footer message. 
        save_all_kpts : bool, optional
            Generate footer text for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints. 
            The default is False. 

        Returns
        -------
        footer_msg : dictionary
            Text string for footer of save file. If save_all_kpts is True,
            generate footer text for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints; else generate footer
            text only for SC kpoints.

        """
        # Generate footer text for SC kpoints
        footer_msg = {}
        footer_msg['SC'] = self._generate_foot_text(self.PBZ_kpts_list_org) if footer_text is None else footer_text
        footer_msg['SpecialKpoints'] = ''
        
        # Save PC, SC kpoints    
        if save_all_kpts:
            footer_msg['PC'] = footer_msg['SC']
            footer_msg['SCPC_map'] = ''
        return footer_msg
    
    def _generate_print_text(self, save_all_kpts:bool=False):
        """
        Create text that will be printed when print information is True.

        Parameters
        ----------
        save_all_kpts : bool, optional
            Generate print message for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints. 
            The default is False. 

        Returns
        -------
        print_msg : dictionary
            Text string for print msg when saving to file. If save_all_kpts is True,
            generate msg text for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints; else generate msg
            text only for SC kpoints.

        """
        # Generate footer text for SC kpoints
        print_msg = {}
        print_msg['SC'] = 'Saving Kpoints to file...'
        print_msg['SpecialKpoints'] = 'Saving special kpoints position indices and labels to file...'
            
        # Save PC, SC kpoints    
        if save_all_kpts:
            print_msg['PC'] = 'Saving kpoints to file...'
            print_msg['SCPC_map'] = 'Saving SC Kpoints - PC kpoints indices mapping to file...'           
        return print_msg

    def _generate_save_contents(self, footer_text=None, save_all_kpts:bool=False):
        """
        Create (kpoints) data that will be saved to file.

        Parameters
        ----------
        footer_text: str, optional
            Footer message. 
        save_all_kpts : bool, optional
            Generate data for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints. 
            The default is False. 

        Returns
        -------
        save_contents_data : dictionary
            Header, footer texts and data to be saved. If save_all_kpts is True,
            generate data for the PC kpoints, generated SC kpoints, 
            SC-PC kpoints mapping, and Special kpoints; else generate data
            only for SC kpoints.

        """
        save_contents_data = {}
        header_msg_ = self._generate_header_text(save_all_kpts=save_all_kpts)
        footer_msg_ = self._generate_footer_text(footer_text=footer_text, save_all_kpts=save_all_kpts)
        print_msg_ = self._generate_print_text(save_all_kpts=save_all_kpts)

        save_contents_data['SC'] = (header_msg_['SC'], self.SBZ_kpts_list, 
                                    footer_msg_['SC'], print_msg_['SC'])  
        save_contents_data['SpecialKpoints'] = (header_msg_['SpecialKpoints'], 
                                                self.special_kpoints_pos_labels, 
                                                footer_msg_['SpecialKpoints'],
                                                print_msg_['SpecialKpoints'])
        if save_all_kpts:
            save_contents_data['PC'] = (header_msg_['PC'], 
                                        self.PBZ_kpts_list_org, 
                                        footer_msg_['PC'],
                                        print_msg_['PC'])
            save_contents_data['SCPC_map'] = (header_msg_['SCPC_map'], 
                                              self.SBZ_PBZ_kpts_mapping, 
                                              footer_msg_['SCPC_map'],
                                              print_msg_['SCPC_map'])
        return save_contents_data

## ============================================================================ 
class BandFolding(_KpointsModule, FindProperties):
    """
    Band folding from primitive to supercell.

    """
    def __init__(self, supercell=None, print_info='low'):
        """
        Initialize the Folding class.

        Parameters
        ----------
        supercell : 3X3 matrix, optional
            Primitive-to-supercell transformation matrix. The default is Identity matrix.
        print_info : [None,'low','medium','high'], optional
            Print information of kpoints folding. Level of printing information. 
            The default is 'low'. If None, nothing is printed.

        """
        if supercell is None: supercell = np.eye(3,dtype=int)
        self.transformation_matrix = _BasicFunctionsModule._check_transformation_matrix(np.array(supercell)) 
        self.print_information = print_info
                   
    def propose_best_least_folding(self, pathPBZ, min_num_pts:int=5, max_num_pts:int=20,
                                   serach_mode:str='brute_force'):
        """
        Calculates SC Kpoints from PC kpoints and returns percent of folding.
        Maximum and Minimum degree of folding are reported.
        
        Folding percent = (#of PC kpoints - #of folded SC Kpoints)/(#of PC kpoints))*100

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

        Returns
        -------
        proposed_folding_results : dictionary
            {index: ((start node, end node), folding data)}
            index : Index of path segment searched from the pathPBZ list supplied.
            folding data : 2d array with each row containing number of division in the 1st
            column and percent of folding in the 2nd column.
            
        """
        return self._propose_max_min_folding(self.transformation_matrix, pathPBZ, 
                                             min_num_pts=min_num_pts, 
                                             max_num_pts=max_num_pts,
                                             serach_mode=serach_mode)
    
    def generate_SC_K_from_pc_k_path(self, pathPBZ=None, nk=11, labels=None, kpts_weights=None, 
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
        assert pathPBZ is not None, 'pathPBZ should not be None.'
        
        # Create PC kpoints from k-path
        PBZ_kpts, special_kpoints_pos_labels, nkgen, labels  = self._generate_kpts_from_kpath(pathPBZ, nk, labels)
        
        # Save SC kpoints
        footer_msg_path = None
        if save_sc_kpts:
            footer_msg_path = self._generate_foot_text(pathPBZ, labels=labels, nk_list=nk)
 
        # Return SC kpoints from PC k-path k-points        
        return self.generate_K_from_k(kpointsPBZ=PBZ_kpts, kpts_weights=kpts_weights, 
                                      save_all_kpts=save_all_kpts,
                                      save_sc_kpts=save_sc_kpts, save_dir=save_dir, 
                                      file_name=file_name, file_name_suffix=file_name_suffix, 
                                      file_format=file_format, footer_msg=footer_msg_path,
                                      special_kpoints_pos_labels=special_kpoints_pos_labels) 
            
    def generate_K_from_k(self, kpointsPBZ=None, kpts_weights=None,
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
        assert kpointsPBZ is not None, 'kpointsPBZ should not be None.'
        
        # Original PC kpoints list
        self.PBZ_kpts_list_org = np.copy(kpointsPBZ)
        # Generate SC kpoints list
        self.PBZ_kpts_list, self.kpointsPBZ_index_in_unique = self._remove_duplicate_kpoints(self.PBZ_kpts_list_org)
        kpointsSBZ = self._find_K_from_k(self.PBZ_kpts_list, self.transformation_matrix)
        self.SBZ_kpts_list, self.kpointsSBZ_index_in_unique = self._remove_duplicate_kpoints(kpointsSBZ)
        self._get_PBZ_SBZ_kpts_mapping()
        
        # Pad the kpoints weights if exists
        if kpts_weights is not None:
            self.SBZ_kpts_list = self._pad_kpts_weights(self.SBZ_kpts_list, kpts_weights=kpts_weights)
        
        # Special kpoints
        self.special_kpoints_pos_labels = special_kpoints_pos_labels
        
        # Print information about folding
        if self.print_information is not None: 
            self._print_info(level=self.print_information)
            
        # Saving kpoints data
        if save_all_kpts or save_sc_kpts:
            # save_contents_data: (header text, data, footer text, print message)
            save_contents_data = self. _generate_save_contents(footer_text=footer_msg, 
                                                               save_all_kpts=save_all_kpts)
            for data_key, data_items in save_contents_data.items():
                SaveData2File.save_sc_kpts_2_file(data=data_items[1],
                                                  save_dir=save_dir, file_name=file_name,
                                                  file_name_suffix=f'{file_name_suffix}_{data_key}', 
                                                  file_format=file_format,
                                                  header_txt=data_items[0], 
                                                  footer_txt=data_items[2],
                                                  print_log=bool(self.print_information),
                                                  print_msg=data_items[3])
        
        return self.PBZ_kpts_list_org, self.PBZ_kpts_list, self.SBZ_kpts_list, \
                self.SBZ_PBZ_kpts_mapping, self.special_kpoints_pos_labels

