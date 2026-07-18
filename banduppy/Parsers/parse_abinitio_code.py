"""
Created on Sat Jul 18 10:54:35 2026

@author: badal.mondal
"""
from ..BasicFunctions.general_functions import _BasicFunctionsModule
from irrep.bandstructure import BandStructure as BandStructure_irrep

#==============================================================================
class _ParseAbInitioCode:
    def __init__(self):
        pass
    
    def _unfold_in_batch_decission(self, wf_file, wf_file_size_cutoff:float=5.0,
                                   wf_file_size_unit:str='GB',
                                   wf_file_pattern:str='*',
                                   wf_n_files:int|float|None=None):
        """
        This function allows to automatically fall back to the batch unfolding
        routine if wave function file is large.

        Parameters
        ----------
        wf_file : file path or str
            Wave function file path.
        wf_file_size_cutoff : float, optional (unit = fsize_unit )
            The cut-off size of the the wave function above which warning msg is
            printed to user requesting to use batch unfolding.
            The default is 5.
        wf_file_size_unit : str, optional ['BYTE','KB','MB','GB','TB']
            Unit of wf_file_size_cutoff. The default is GB.
        wf_file_pattern : str or None, optional
            Glob pattern to find specific type of files. E.g. 'wfc*.' for qe etc.
            Only important when wf_file is directory.
            The default is * == all files in the directory. 
         wf_n_files : int or float or None, optional
             This will multiplied by a single file size in conditional checking.
             This number could be number of K-points to read for qe code for e.g.

        Returns
        -------
        None
        """      
        try:
            file_size_found, file_too_large = \
                _BasicFunctionsModule._check_file_size(wf_file, _file_size_cutoff=wf_file_size_cutoff,
                                                       _file_size_unit=wf_file_size_unit,
                                                       _file_pattern=wf_file_pattern,
                                                       _n_files=wf_n_files)
            if file_too_large and (not self.kpts_batch_unfold_): 
                print(f"USER RECOMMENDATION: Total wavefunction file size is estimated ~ {file_size_found:0.2f} {wf_file_size_unit}.")
                print(f"Since, this is >{wf_file_size_cutoff} {wf_file_size_unit}, if memory error occures we recommend the user")
                print("to set unfold_kpts_in_batch=True in Unfold().")
                #print(f"WARNING: Wavefunction is large. Falling back to batch unfolding with default kpt_batch_size")
                #self.kpts_batch_unfold_ = True
        except:
            print("WARNING: Can't check wave function file size. Can't provide special user recommendation. \
                  Do not worry though! Wouldn't affect unfolding results.")
    
    def _read_irrep_vasp(self, **kwargs):
        self._unfold_in_batch_decission(self.vasp_kwards_['wavecar_file_path'],
                                        wf_file_size_cutoff=5.0, 
                                        wf_file_size_unit='GB')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        from .parse_vasprunxml import VasprunXml
       
        if (self.fermi_energy_ is None) or self.zero_weight_kp_:
            vr = VasprunXml(self.vasp_kwards_['vasprunxml_file_path'])  
            
        if self.fermi_energy_ is None: 
            self.fermi_energy_ = vr.vasprun_data['efermi']
            
        if self.zero_weight_kp_:
            self.only_unfold_for_kpts_idxs_ = self._get_zero_weight_kp_indices(vr.vasprun_data['kpoints_wts'])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return BandStructure_irrep.from_vasp(fWAV=self.vasp_kwards_['wavecar_file_path'], 
                                             fPOS=self.vasp_kwards_['poscar_file_path'],
                                             spinor=self.vasp_kwards_['is_spin_nondegenrate'],
                                             spin_channel=self.vasp_kwards_['unfold_spin_channel'],
                                             Ecut=self.vasp_kwards_['wf_cutoff_energy'], 
                                             EF=self.fermi_energy_,
                                             IBstart=self.only_unfold_band_idx_[0],
                                             IBend=self.only_unfold_band_idx_[1],
                                             kplist=self.only_unfold_for_kpts_idxs_,
                                             read_kpoints=not self.kpts_batch_unfold_,
                                             verbosity=self.print_log_,
                                             onlysym=self.onlysym_, 
                                             irreps=self.irreps_,  
                                             spacegroup=self.spacegroup_,
                                             **kwargs)
    
    def _read_irrep_qe(self, **kwargs):
        _prefix = f"{self.qe_kwards_['output_file_dir']}/{self.qe_kwards_['save_file_prefix']}"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        noofkpts = len(self.only_unfold_for_kpts_idxs_) if self.only_unfold_for_kpts_idxs_ else None
        self._unfold_in_batch_decission(f"{_prefix}.save", wf_file_size_cutoff=5.0, 
                                        wf_file_size_unit='GB', wf_file_pattern='wfc*.*',
                                        wf_n_files=noofkpts)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return BandStructure_irrep.from_espresso(prefix=_prefix, alat=None,
                                                 spin_channel=self.qe_kwards_['unfold_spin_channel'],
                                                 Ecut=self.qe_kwards_['wf_cutoff_energy'], 
                                                 EF=self.fermi_energy_,
                                                 IBstart=self.only_unfold_band_idx_[0],
                                                 IBend=self.only_unfold_band_idx_[1],
                                                 kplist=self.only_unfold_for_kpts_idxs_,
                                                 read_kpoints=not self.kpts_batch_unfold_,
                                                 verbosity=self.print_log_,
                                                 onlysym=self.onlysym_, 
                                                 irreps=self.irreps_,  
                                                 spacegroup=self.spacegroup_,
                                                 **kwargs)
    
    def _read_irrep_abinit(self, **kwargs):
        self._unfold_in_batch_decission(self.abinit_kwards_['wfk_file_path'],
                                        wf_file_size_cutoff=5.0, 
                                        wf_file_size_unit='GB')
        return BandStructure_irrep.from_abinit(fWFK=self.abinit_kwards_['wfk_file_path'], 
                                               spin_channel=self.abinit_kwards_['unfold_spin_channel'],
                                               Ecut=self.abinit_kwards_['wf_cutoff_energy'], 
                                               EF=self.fermi_energy_,
                                               IBstart=self.only_unfold_band_idx_[0],
                                               IBend=self.only_unfold_band_idx_[1],
                                               kplist=self.only_unfold_for_kpts_idxs_,
                                               read_kpoints=not self.kpts_batch_unfold_,
                                               verbosity=self.print_log_,
                                               onlysym=self.onlysym_, 
                                               irreps=self.irreps_,  
                                               spacegroup=self.spacegroup_,
                                               **kwargs)
    
    def _read_irrep_gpaw(self, **kwargs):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO: Check wave function file size. Note irrep uses
        # gpaw calculator instance here to read wave functions.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return BandStructure_irrep.from_gpaw(calculator_gpaw=self.gpaw_kwards_['gpaw_calculator_instance'], 
                                             read_paw=self.gpaw_kwards_['read_paw'],
                                             spinor=self.gpaw_kwards_['is_spin_nondegenrate'],
                                             spin_channel=self.gpaw_kwards_['unfold_spin_channel'],
                                             Ecut=self.gpaw_kwards_['wf_cutoff_energy'], 
                                             EF=self.fermi_energy_,
                                             IBstart=self.only_unfold_band_idx_[0],
                                             IBend=self.only_unfold_band_idx_[1],
                                             kplist=self.only_unfold_for_kpts_idxs_,
                                             read_kpoints=not self.kpts_batch_unfold_,
                                             verbosity=self.print_log_,
                                             onlysym=self.onlysym_, 
                                             irreps=self.irreps_,  
                                             spacegroup=self.spacegroup_,
                                             **kwargs)
    
    
    def _read_irrep_wannier90(self, **kwargs):
        _prefix = f"{self.wannier90_kwards_['output_file_dir']}/{self.wannier90_kwards_['seedname']}"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        noofkpts = len(self.only_unfold_for_kpts_idxs_) if self.only_unfold_for_kpts_idxs_ else None
        self._unfold_in_batch_decission(_prefix, 
                                        wf_file_size_cutoff=5.0, 
                                        wf_file_size_unit='GB', 
                                        wf_file_pattern='UNK*.*',
                                        wf_n_files=noofkpts)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return BandStructure_irrep.from_wannier90(prefix=_prefix, 
                                                  unk_formatted=self.wannier90_kwards_['input_files_are_text_format'],
                                                  spinor=self.wannier90_kwards_['is_spin_nondegenrate'],
                                                  spin_channel=self.wannier90_kwards_['unfold_spin_channel'],
                                                  Ecut=self.wannier90_kwards_['wf_cutoff_energy'], 
                                                  EF=self.fermi_energy_,
                                                  IBstart=self.only_unfold_band_idx_[0],
                                                  IBend=self.only_unfold_band_idx_[1],
                                                  kplist=self.only_unfold_for_kpts_idxs_,
                                                  read_kpoints=not self.kpts_batch_unfold_,
                                                  verbosity=self.print_log_,
                                                  onlysym=self.onlysym_, 
                                                  irreps=self.irreps_,  
                                                  spacegroup=self.spacegroup_,
                                                  **kwargs)