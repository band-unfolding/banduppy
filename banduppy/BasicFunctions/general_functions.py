import numpy as np
import pickle

_draw_line_length = 72

## ============================================================================   
class _BasicFunctionsModule(object):
    
    @staticmethod
    def _check_transformation_matrix(M, tolerance = 1e-14):
        """
        Check shape and commensurate property of PC-SC transformation matrix.

        Parameters
        ----------
        M : ndarray
            Matrix to check.
        tolerance : float, optional
            Tolerance to check if transformation matrix is commensurate. The default is 1e-14.

        Returns
        -------
        M_f : ndarray of int
            Transformation matrix.

        """
        assert M.shape==(3,3), "supercell should be 3x3, found {}".format(M.shape)
        
        M_int = np.round(M)
        assert np.linalg.norm(np.array(M) - M_int) < tolerance , "supercell should consist of integers, found {}".format(M)
        
        M_f = np.array(M_int,dtype=int)
        assert np.linalg.det(M_f) != 0, "the supercell vectors should be linear independent"
        
        return M_f
    
    @staticmethod
    def _round_2_tolerance(input_array, decimals=10):
        '''
        Use rounding to avoid flotting point precision.
        o 0.1%1 != 1.1%1
        o 0.1%1 == np.round(1.1%1, decimals=10)
        '''
        return np.round(input_array, decimals=decimals)
    
    @staticmethod
    def _return_mod(input_array, round_decimals:bool=True):
        """
        Return mod value. => Scale k-points.

        Parameters
        ----------
        input_array : numpy ndarray
            Input array to round decimals.
        round_decimals : bool, optional
            Round decimals or not? The default is True.

        Returns
        -------
        numpy ndarray
            After rounding the float decimal points.

        """
        if round_decimals:
            return _BasicFunctionsModule._round_2_tolerance(input_array%1)
        else:
            return input_array%1
        
## ============================================================================
class SaveData2File:
    def __init__(self):
        pass
    
    @staticmethod
    def default_save_settings(save_data):
        tmp_save = {'save2file': False, 'fdir': '.', 'fname': 'test', 'fname_suffix': ''}
        for ll in tmp_save:
            if ll in save_data:
                tmp_save[ll] = save_data[ll]
        return tmp_save
    
    @staticmethod
    def save_2_file(data=None, save_dir='.', file_name:str='', file_name_suffix:str='', 
                    header_txt:str='', footer_txt:str='',comments_symbol='! ',
                    np_data_fmt='%12.8f', print_log:bool=True):
        """
        Save the generated SC kpoints to a file.

        Parameters
        ----------
        data : numpy array, optional
            Data to be saved. The default is None.
        save_dir : str/path_object, optional
            Directory to save the file. The default is current directory.
        file_name : str, optional
            Name of the file. The default is ''.
            If file_format is vasp, file_name=KPOINTS_<file_name_suffix>
        file_name_suffix : str, optional
            Suffix to add after the file_name. The default is ''.
        header_txt : str, optional
            String that will be written at the beginning of the file. The default is None.
        footer_txt : str, optional
            String that will be written at the end of the file.. The default is None.
        comments_symbol : str, optional
            String that will be prepended to the header and footer strings, 
            to mark them as comments. The default is ‘!‘. 
        np_data_fmt: str
            Data format for numpy.savetxt.
        print_log : bool, optional
            Print path of save file. The default is False.

        Returns
        -------
        String/path object
            File path where the data is saved.

        """
        if data is None: return
        fname_save_file = f'{save_dir}/{file_name}{file_name_suffix}'
        if isinstance(data, np.ndarray):
            with open(fname_save_file, 'w') as f:
                np.savetxt(f, data, header=header_txt, footer=footer_txt, 
                           fmt=np_data_fmt, comments=comments_symbol)
        elif isinstance(data, dict):
            fname_save_file += '.pkl'
            with open(fname_save_file, 'wb') as f:
                # Note: the header and footer msg are not saved for the time being.
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return fname_save_file
    
    @staticmethod
    def save_sc_kpts_2_file(data=None, save_dir='.', file_name:str='', 
                            file_name_suffix:str='', file_format:str='vasp',
                            header_txt:str='', footer_txt:str='',comments_symbol='! ',
                            print_log:bool=False, print_msg:str='Saving to file...'):
        """
        Save the generated SC kpoints to a file.

        Parameters
        ----------
        data : numpy array, optional
            Data to be saved. The default is None.
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
        header_txt : str, optional
            String that will be written at the beginning of the file. The default is None.
        footer_txt : str, optional
            String that will be written at the end of the file.. The default is None.
        comments_symbol : str, optional
            String that will be prepended to the header and footer strings, 
            to mark them as comments. The default is ‘!‘. 
        print_log : bool, optional
            Print path of save file. The default is False.
        print_msg : str, optional
            Message to print on print_log. The default is ''.

        Returns
        -------
        String/path object
            File path where the data is saved.

        """
        if file_format == 'vasp':
            file_name = 'KPOINTS'
            comments_symbol = ''
            
        if print_log: print(f"{'='*_draw_line_length}\n- {print_msg}.")
        
        fname_save_file = \
        SaveData2File.save_2_file(data=data, save_dir=save_dir, file_name=file_name, 
                                  file_name_suffix=file_name_suffix, header_txt=header_txt, 
                                  footer_txt=footer_txt,comments_symbol=comments_symbol)
        if print_log: print(f'-- Filepath: {fname_save_file}\n- Done')
        return fname_save_file