"""
Created on Wed Jul  8 17:48:54 2026

@author: badal.mondal
"""

#import warnings
import numpy as np
from .parse_abinitio_code import _ParseAbInitioCode

### ===========================================================================  

class _ProcessPWs(_ParseAbInitioCode):
    def __init__(self, ab_initio_code:str='vasp', 
                 only_unfold_for_kpts_idxs:np.ndarray|list[int]|None=None,
                 only_unfold_band_idx:tuple[int|None, int|None]|list[int | 
                 None, int|None] = (None, None), kpts_batch_unfold:bool=False, 
                 zero_weight_kp:bool=False, fermi_energy:float|None=None,
                 vasp_kwards:dict|None = None, qe_kwards:dict|None=None,  
                 abinit_kwards:dict|None=None, gpaw_kwards:dict|None=None,
                 wannier90_kwards:dict|None=None,print_log=None):
        """

        Parameters
        ----------
        ab_initio_code : str, optional
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
        kpts_batch_unfold : bool, optional
            Whether to unfold complete wave function file in sigle shot. If False,
            unfolding is performed in K-point batch. Chunk of K-point info is loaded in RAM, 
            unfolded, and then removed from RAM once unfolding is done. This allows 
            us to aviod large memory requirement when the wave function file is 
            very large (~few GB). The default is False.            
        zero_weight_kp : bool, optional
            Whether the K-points in ab-initio calculation corresponds to zero-weight 
            k-point method. This automatically exclude unfolding of non-zero weight
            K-points. The default is False.
        fermi_energy : float|None, optional
            User supplied Fermi-energy. If None, by default it is extracted from
            output files corresponds to specific ab-inito codes. The default is None.
        vasp_kwards : dict | None, optional
            The keywards specific to VASP ab-initio code. Will be ignored when 
            ab_init_code != vasp. The default is None.
        qe_kwards : dict | None, optional
            The keywards specific to Quantum ESPRESSO ab-initio code. Will be 
            ignored when ab_init_code != qe. The default is None.
        abinit_kwards : dict | None, optional
            The keywards specific to ABINIT ab-initio code. Will be ignored when 
            ab_init_code != abinit. The default is None.
        gpaw_kwards : dict | None, optional
            The keywards specific to GPAW ab-initio code. Will be ignored when 
            ab_init_code != gpaw. The default is None.
        wannier90_kwards : dict | None, optional
            The keywards specific to WANNIER90 ab-initio code. Will be ignored when 
            ab_init_code != wannier90. The default is None.
        print_log : [None,'low','medium','high'], optional
            Level of printing information. If None, nothing is printed.
            The default is 'low'. 

        Returns
        -------
        None
        """
        _irrep_verbosity_map = {None: 0, 'low': 1, 'medium':2, 'high':3}
        self.print_log_ = _irrep_verbosity_map[print_log]
        self.ab_initio_code_ = ab_initio_code.lower()
        self.zero_weight_kp_ = zero_weight_kp
        self._check_ab_inito_code_related_conditions()
        
        self.only_unfold_band_idx_ = only_unfold_band_idx
        self.only_unfold_for_kpts_idxs_ = only_unfold_for_kpts_idxs
        self.kpts_batch_unfold_ = kpts_batch_unfold
        self.fermi_energy_ = fermi_energy

        self.vasp_kwards_ = self._check_reset_vasp_keywards(vasp_kwards)
        self.qe_kwards_ = self._check_reset_qe_keywards(qe_kwards)
        self.abinit_kwards_ = self._check_reset_abinit_keywards(abinit_kwards)
        self.gpaw_kwards_ = self._check_reset_gpaw_keywards(gpaw_kwards)
        self.wannier90_kwards_ = self._check_reset_wannier90_keywards(wannier90_kwards)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Some of the irrep keywards are set by banduppy, user does not have access
        self.onlysym_ = False 
        self.irreps_ = False 
        self.spacegroup_ = None
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _ParseAbInitioCode.__init__(self)
    
    def _check_ab_inito_code_related_conditions(self):
        _ab_code_implemented = ['vasp', 'qe', 'espresso', 'quantum_espresso', 
                                'abinit', 'gpaw', 'wannier90'] 
        code_not_support_msg = f'''{self.ab_initio_code_} is not supported in banduppy yet. Contact developer. 
        Allowed codes: {_ab_code_implemented}. '''
           
        zero_weight_kp_method_support_code = ['vasp']
        zero_weightcode_not_support_msg = f'''0-weight kpoint method is not available for {self.ab_initio_code_}. 
        Contact developer. Only available for {zero_weight_kp_method_support_code}'''
        
        if self.ab_initio_code_ not in _ab_code_implemented:
            raise ValueError(code_not_support_msg)
            
        if self.zero_weight_kp_ and (self.ab_initio_code_ not in zero_weight_kp_method_support_code):
            raise ValueError(zero_weightcode_not_support_msg)
                
    def _generate_bandstructure_instance(self, **kwargs):
        if self.ab_initio_code_ == 'vasp':
            return self._read_irrep_vasp(**kwargs)
        elif self.ab_initio_code_ in ['qe', 'espresso', 'quantum_espresso']:
            return self._read_irrep_qe(**kwargs)
        elif self.ab_initio_code_ == 'abinit':
            return self._read_irrep_abinit(**kwargs)
        elif self.ab_initio_code_ == 'gpaw':
            return self._read_irrep_gpaw(**kwargs)
        elif self.ab_initio_code_ == 'wannier90':
            return self._read_irrep_wannier90(**kwargs)
        else:
            return None
    
    @classmethod        
    def _check_reset_vasp_keywards(cls, vasp_keywards:dict):
        """
        Check user specified VASP code related keywards. If not any dictionary key
        is not found, will be reset to default.

        Parameters
        ----------
        vasp_keywards : dict
            Following (key, value) pairs are allowed:
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

        Returns
        -------
        vasp_keywards : dict
            Default keywards adjusted to user supplied values.
        """
        default_vasp_kwards = {'poscar_file_path': './POSCAR', 
                               'wavecar_file_path':'./WAVECAR', 
                               'vasprunxml_file_path': './vasprun.xml',
                               'wf_cutoff_energy': None,
                               'is_spin_nondegenrate': False, 
                               'unfold_spin_channel': None}
        return cls._reset_abinitio_code_keywards(default_vasp_kwards, vasp_keywards)
    
    @classmethod        
    def _check_reset_qe_keywards(cls, qe_keywards:dict):
        """
        Check user specified Quantum Espresso code related keywards. If not any dictionary key
        is not found, will be reset to default.

        Parameters
        ----------
        qe_keywards : dict
            Following (key, value) pairs are allowed:
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

        Returns
        -------
        qe_keywards : dict
            Default keywards adjusted to user supplied values.
        """
        default_qe_kwards = {'output_file_dir': './', 
                             'save_file_prefix':'prefix', 
                             'wf_cutoff_energy': None,
                             'unfold_spin_channel': None}
        return cls._reset_abinitio_code_keywards(default_qe_kwards, qe_keywards)
    
    @classmethod        
    def _check_reset_abinit_keywards(cls, abinit_keywards:dict):
        """
        Check user specified ABINIT code related keywards. If not any dictionary key
        is not found, will be reset to default.

        Parameters
        ----------
        abinit_keywards : dict
            Following (key, value) pairs are allowed:
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

        Returns
        -------
        abinit_keywards : dict
            Default keywards adjusted to user supplied values.
        """
        default_abinit_kwards = {'wfk_file_path': './test_WFK', 
                                 'wf_cutoff_energy': None,
                                 'unfold_spin_channel': None}
        return cls._reset_abinitio_code_keywards(default_abinit_kwards, abinit_keywards)

    @classmethod        
    def _check_reset_gpaw_keywards(cls, gpaw_keywards:dict):
        """
        Check user specified GPAW code related keywards. If not any dictionary key
        is not found, will be reset to default.

        Parameters
        ----------
        gpaw_keywards : dict
            Following (key, value) pairs are allowed:
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

        Returns
        -------
        gpaw_keywards : dict
            Default keywards adjusted to user supplied values.
        """
        default_gpaw_kwards = {'gpaw_calculator_instance': None, 
                               'read_paw': False, 
                               'is_spin_nondegenrate':False,
                               'wf_cutoff_energy': None,
                               'unfold_spin_channel': None}
        return cls._reset_abinitio_code_keywards(default_gpaw_kwards, gpaw_keywards)
    
    @classmethod        
    def _check_reset_wannier90_keywards(cls, wannier90_keywards:dict):
        """
        Check user specified WANNIER90 code related keywards. If not any dictionary key
        is not found, will be reset to default.

        Parameters
        ----------
        wannier90_keywards : dict
            Following (key, value) pairs are allowed:
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

        Returns
        -------
        wannier90_keywards : dict
            Default keywards adjusted to user supplied values.
        """
        default_wannier90_kwards = {'output_file_dir': './', 
                                    'seedname':'prefix', 
                                    'input_files_are_text_format':False,
                                    'is_spin_nondegenrate':False,
                                    'wf_cutoff_energy': None,
                                    'unfold_spin_channel': None}
        return cls._reset_abinitio_code_keywards(default_wannier90_kwards, wannier90_keywards)
    
    @staticmethod    
    def _reset_abinitio_code_keywards(default_kwards:dict, code_keywards:dict|None=None):
        if code_keywards is None:
            return default_kwards
        
        #default_keywords = default_kwards.copy()
        for ll in default_kwards:
            if ll in code_keywards:
                default_kwards[ll] = code_keywards[ll]
        return default_kwards
    
    @staticmethod
    def _get_zero_weight_kp_indices(ibz_kpoints_wts):
        """
        This function returns indices of the zero weight k-points.

        Parameters
        ----------
        ibz_kpoints_wts : list/array of float
            k-point weights as read from the k-point file.
            For VASP: vasprun.xml file is read.

        Returns
        -------
        list of int
            Indices of the zero weight k-points.
        """
        zero_weight_indices = np.nonzero(ibz_kpoints_wts < 1e-6)[0]
        if len(zero_weight_indices) > 0:
            unfold_for_kpts_inds = zero_weight_indices
        else:
            print('WARNING: No 0-weight kpoints are found. Ignoring 0-weight related tags. Unfolding for all avable k-points.')
            unfold_for_kpts_inds = None
        return unfold_for_kpts_inds