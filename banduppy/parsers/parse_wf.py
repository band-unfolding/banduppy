"""
Created on Wed Jul  8 17:48:54 2026

@author: badal.mondal
"""

#import warnings
import numpy as np
from irrep.bandstructure import BandStructure as BandStructure_irrep
from ..BasicFunctions.general_functions import _BasicFunctionsModule

### ===========================================================================  

class _ProcessPWs(BandStructure_irrep):
    
    def __init__(self, ab_initio_code:str='vasp', wf_cutoff_energy:float|None=None, 
                 only_unfold_for_kpts_idxs:np.ndarray|list[int]|None=None,
                 only_unfold_band_idx:tuple[int|None, int|None]|list[int | 
                 None, int|None] = (None, None), kpts_batch_unfold:bool=False, 
                 zero_weight_kp:bool=False, fermi_energy:float|None=None,
                 vasp_kwards = None, abinit_kwards=None,
                 qe_kwards=None, gpaw_kwards=None,
                 wannier90_kwards=None,print_log=None):
        _irrep_verbosity_map = {None: 0, 'low': 1, 'medium':2, 'high':3}
        self.print_log_ = _irrep_verbosity_map[print_log]
        self.ab_initio_code_ = ab_initio_code
        self.wf_cutoff_energy_ = wf_cutoff_energy
        self.only_unfold_band_idx_ = only_unfold_band_idx
        self.only_unfold_for_kpts_idxs_ = only_unfold_for_kpts_idxs
        self.kpts_batch_unfold_ = kpts_batch_unfold
        self.zero_weight_kp_ = zero_weight_kp
        self.fermi_energy_ = fermi_energy
        self.vasp_kwards_ = vasp_kwards
        self.abinit_kwards_ = abinit_kwards
        self.qe_kwards_ = qe_kwards
        self.gpaw_kwards_ = gpaw_kwards
        self.wannier90_kwards_ = wannier90_kwards

    def _unfold_in_batch_decission(self, wf_file, wf_file_size_cutoff:float=5.0):
        """
        This function allows to automatically fall back to the batch unfolding
        routine if wave function file is large.

        Parameters
        ----------
        wf_file : file path or str
            Wave function file path.
        wf_file_size_cutoff : float (unit: GB)
            The cut-off size of the the wave function above which warning msg is
            printed to user requesting to use batch unfolding.
            The default is 5 GB.

        Returns
        -------
        None
        """
        file_too_large = _BasicFunctionsModule._check_file_size(wf_file, 'GB') > 5
        if file_too_large and (not self.kpts_batch_unfold_): 
            print(f"WARNING: Wavefunction file is >{wf_file_size_cutoff} GB. If memory error occure, set unfold_kpts_in_batch=True in Unfold().")
            #print(f"WARNING: Wavefunction is large. Falling back to batch unfolding with default kpt_batch_size")
            #self.kpts_batch_unfold_ = True
                
    def _generate_bandstructure_instance(self, **kwargs):
        _ab_code_implemented = ['vasp', 'qe', 'espresso', 'quantum_espresso', 
                                'abinit', 'gpaw', 'wannier90'] 
        code_not_support_msg = f'''{self.ab_initio_code_} is not supported in banduppy yet. Contact developer. 
        Allowed codes: {_ab_code_implemented}. '''
           
        zero_weight_kp_method_support_code = ['vasp']
        zero_weightcode_not_support_msg = f'''0-weight kpoint method is not available for {self.ab_initio_code_}. 
        Contact developer. Only available for {zero_weight_kp_method_support_code}'''
        
        #//////////////////////////////////////////////////////////////////////
        if self.ab_initio_code_ not in _ab_code_implemented:
            raise ValueError(code_not_support_msg)
            
        if self.zero_weight_kp_ and (self.ab_initio_code_ not in zero_weight_kp_method_support_code):
            raise ValueError(zero_weightcode_not_support_msg)
        
        #//////////////////////////////////////////////////////////////////////
        if self.ab_initio_code_ == 'vasp':
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            from .parse_vasprunxml import VasprunXml
            vr = VasprunXml(self.vasp_kwards_['vasprunxml_file_path'])            
            if self.fermi_energy_ is None: 
                self.fermi_energy_ = vr.vasprun_data['efermi']
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.zero_weight_kp_:
                self.only_unfold_for_kpts_idxs_ = self._get_zero_weight_kp_indices(vr.vasprun_data['kpoints_wts'])
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self._unfold_in_batch_decission(self.vasp_kwards_['wavecar_file_path'])
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return BandStructure_irrep.from_vasp(fWAV=self.vasp_kwards_['wavecar_file_path'], 
                                                 fPOS=self.vasp_kwards_['poscar_file_path'],
                                                 spinor=self.vasp_kwards_['is_spin_nondegenrate'],
                                                 spin_channel=self.vasp_kwards_['unfold_spin_channel'],
                                                 Ecut=self.wf_cutoff_energy_, 
                                                 EF=self.fermi_energy_,
                                                 IBstart=self.only_unfold_band_idx_[0],
                                                 IBend=self.only_unfold_band_idx_[1],
                                                 kplist=self.only_unfold_for_kpts_idxs_,
                                                 read_kpoints=not self.kpts_batch_unfold_,
                                                 verbosity=self.print_log_,
                                                 onlysym=False, irreps=False,  
                                                 spacegroup=None,**kwargs)
        #//////////////////////////////////////////////////////////////////////
        elif self.ab_initio_code_ in ['qe', 'espresso', 'quantum_espresso']:
            pass
        #//////////////////////////////////////////////////////////////////////
        elif self.ab_initio_code_ == 'abinit':
            pass
        #//////////////////////////////////////////////////////////////////////
        elif self.ab_initio_code_ == 'gpaw':
            pass
        #//////////////////////////////////////////////////////////////////////
        elif self.ab_initio_code_ == 'wannier90':
            pass
        #//////////////////////////////////////////////////////////////////////
        else:
            raise ValueError(code_not_support_msg)
            
        
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