# ---------------------------- Import modules ---------------------------------
import numpy as np
import pickle
import banduppy

print(f'- BandUPpy version: {banduppy.__version__}')

#%% ------------------------ Define variables ---------------------------------
# supercell dimension 
## A general note on supercell dimension: If you have let say a 4X4X2 supercell then you can explicitely write 
## super_cell_size = [[4, 0, 0], [0, 4, 0], [0, 0, 2]] or you can also simply pass
## super_cell_size = np.diag([4,4,2]). Both was defining super_cell_size is allowed.
super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] # this is for our Si 8-atom conventional unitcell/supercell case. Not to be confused with above note.
# k-path: L-G-X-U,K-G. If the segmant is skipped, put a None between nodes.
PC_BZ_path = [[1/2,1/2,1/2], [0,0,0],[1/2,0,1/2], [5/8,1/4,5/8], None, [3/8,3/8,3/4], [0,0,0]] 
# Number of k-points in each path segments. or, one single number if they are same.
npoints_per_path_seg = (21,21,6,21) 
# Labels of special k-points: list or string. e.g ['L','G','X','U','K','G'] or 'LGXUKG'
special_k_points = "LGXUKG"
# Weights of the k-points to be appended in the final generated k-points files
kpts_weights = 1 
# Save the SC kpoints in a file
save_to_file = True 
# Directory to save file
save_to_dir = '<directory to save files>' 
# File format of kpoints file that will be created and saved
kpts_file_format = 'vasp' # This will generate vasp KPOINTS file format

#%% ---------------------- Initiate Unfolding method --------------------------
band_unfold = banduppy.Unfolding(supercell=super_cell_size, print_log='high')

#%% ------------ Creating SC folded kpoints from PC band path -----------------
kpointsPBZ_full, kpointsPBZ_unique, kpointsSBZ, \
SBZ_PBZ_kpts_mapping, special_kpoints_pos_labels \
= band_unfold.generate_SC_Kpts_from_pc_k_path(pathPBZ = PC_BZ_path,
                                              nk = npoints_per_path_seg,
                                              labels = special_k_points,
                                              kpts_weights = kpts_weights,
                                              save_kpts = save_to_file,
                                              save_dir = save_to_dir,
                                              file_name_suffix = '',
                                              file_format=kpts_file_format)

#%% ------------------------ Read wave function file --------------------------
print(f"{'='*72}\n - Unfolding bands...")
read_dir = '<path where the vasp output files are>'
bands = banduppy.BandStructure(code="vasp", spinor=False, 
                               fPOS = f"{read_dir}/POSCAR",
                               fWAV = f"{read_dir}/WAVECAR")

#%% ----------------- Unfold the band structures ------------------------------
# save2file : Save unfolded kpoints or not? 
# fdir : Directory path where to save the file.
# fname : Name of the file.
# fname_suffix : Suffix to add to the file name.

# Option 1: Continue with previous instance.
unfolded_bandstructure_, kpline \
= band_unfold.Unfold(ab_initio_code='vasp', 
                     unfold_kpts_in_batch=False, 
                     #unfold_kpts_in_batch=True, kpt_batch_size=10, # Memory efficient routine
                     vasp_keywards = {'poscar_file_path': f'{read_dir}/POSCAR', 
                                      'wavecar_file_path':f'{read_dir}/WAVECAR', 
                                      'vasprunxml_file_path': f'{read_dir}/vasprun.xml',
                                      'is_spin_nondegenrate': False, 
                                      'unfold_spin_channel': None # ['up', 'dw']
                                      },
                     kline_discontinuity_threshold = 0.1, 
                     save_unfolded_kpts = {'save2file': True, 
                                           'fdir': save_to_dir,
                                           'fname': 'kpoints_unfolded',
                                           'fname_suffix': ''},
                     save_unfolded_bandstr = {'save2file': True, 
                                              'fdir': save_to_dir,
                                              'fname': 'bandstructure_unfolded',
                                              'fname_suffix': ''})

# Option 2: If this part is used independently from the above instances, 
# re-initiate the Unfolding module.
# # --------------------- Initiate Unfolding method --------------------------
# band_unfold = banduppy.Unfolding(supercell=super_cell_size,
#                                  print_info='high')
# # ----------------- Unfold the band structures ------------------------------
# unfolded_bandstructure_, kpline \
# = band_unfold.Unfold(ab_initio_code='vasp', 
#                      unfold_kpts_in_batch=False, 
#                      #unfold_kpts_in_batch=True, kpt_batch_size=10, # Memory efficient routine
#                      vasp_keywards = {'poscar_file_path': f'{read_dir}/POSCAR', 
#                                       'wavecar_file_path':f'{read_dir}/WAVECAR', 
#                                       'vasprunxml_file_path': f'{read_dir}/vasprun.xml',
#                                       'is_spin_nondegenrate': False, 
#                                       'unfold_spin_channel': None # ['up', 'dw']
#                                       }, 
#                      PBZ_kpts_list_full=kpointsPBZ_full, 
#                      SBZ_kpts_list=kpointsSBZ, 
#                      SBZ_PBZ_kpts_map=SBZ_PBZ_kpts_mapping,
#                      kline_discontinuity_threshold = 0.1, 
#                      save_unfolded_kpts = {'save2file': True, 
#                                            'fdir': save_to_dir,
#                                            'fname': 'kpoints_unfolded',
#                                            'fname_suffix': ''},
#                      save_unfolded_bandstr = {'save2file': True, 
#                                               'fdir': save_to_dir,
#                                               'fname': 'bandstructure_unfolded',
#                                               'fname_suffix': ''})

#%% ---------------- Determine band centers and band width --------------------
# -------------------- Initiate Properties method -----------------------------
unfolded_band_properties = banduppy.Properties(print_log='high')
#===================================
#######################
# For 'Mondal2025', most important parameters to tune for good results are
# gaussian_decay_sigma, min_dN, min_sum_dNs_for_a_band. However, based on my experience 
# the default values are already good for huge number of systems I have tested.
#######################
# Algorithm to determine band centers and broadening. 
# Options are: 'Medeiros2014','Mondal2025'
# The default is 'Mondal2025'.
# Note: 
# 1. Medeiros2014 algorithm uses dN as weights during band center determination.
# o average_bandcenter = average(energies, weights=dN)
# 2. Mondal2025 version additionally uses distance weights (Gaussian decay 
# around average reference band center).
# o distace_2_ref = average(energies, weights=dN)
# o weights_distance = exp(-(distace_2_ref**2)/(2*sigma**2))
# o average_bandcenter = average(energies, weights=dN*weights_distance) 
bandcenter_algorithm_ = 'Mondal2025' 
# Standard deviation of Gaussian decay for the weights during average band center
# determination. The decay is based on energy axis. The default is 1. The
# default works well most often.
# Note: This parameter is used in 'Mondal2025' algorithm_version. 
# o distace_2_ref = average(energies, weights=dN)
# o weights_distance = exp(-(distace_2_ref**2)/(2*sigma**2))
# o average_bandcenter = average(energies, weights=dN*weights_distance)
gaussian_decay_sigma = 1
# Discard the bands which has weights below min_dN to start with. 
# This pre-screening step helps to minimize the data that will processed.
# This parameter just pre-screen/minimize amount of data that will be passed to
# band center determination SCF algorithm and independent of min_sum_dNs_for_a_band parameter.
min_dN = 1e-2
# Cut off criteria for minimum weights that a band center should have. 
# The band centers with lower weights than min_sum_dNs_for_a_band will be
# discarded during SCF refinements. If min_sum_dNs_for_a_band  
# is smaller than threshold_dN_2b_trial_band_center, min_sum_dNs_for_a_band
# will be reset to threshold_dN_2b_trial_band_center value.
min_sum_dNs_for_a_band = 0.1 
#===================================
# Initial guess of the band centers based on the threshold wights.
threshold_dN_2b_trial_band_center = None
# The tolerance to group the bands set per unique kpoints value. This determines if two
# flotting point numbers are same or not. This is not a critical parameter for 
# band center determination algorithm.
err_tolerance = 1e-8
# Precision when compared band centers from previous and current SCF
# iteration. SCF is considered converged if this precision is reached.
prec_pos_band_centers = 1e-5 # in eV
#===================================
unfolded_bandstructure_properties, all_scf_data = \
    unfolded_band_properties.band_centers_broadening_bandstr(unfolded_bandstructure_, 
                                                             algorithm_version=bandcenter_algorithm_,
                                                             sigma=gaussian_decay_sigma,
                                                             min_sum_dNs_for_a_band=min_sum_dNs_for_a_band,
                                                             min_dN_pre_screening=min_dN,
                                                             threshold_dN_2b_trial_band_center=
                                                             threshold_dN_2b_trial_band_center,
                                                             precision_scf_band_centers=prec_pos_band_centers,
                                                             err_tolerance_compare_kpts_val=err_tolerance,
                                                             collect_scf_data=False)

#%% --------------------- Plot band structure ---------------------------------
# Fermi energy
Efermi = None #5.9740
# Minima in Energy axis to plot
Emin = -5
# Maxima in Energy axis to plot
Emax = 5
# Filename to save the figure. If None, figure will not be saved
save_file_name = 'unfolded_bandstructure.png'

# Option 1: Continue with previous instance.    
fig, ax, CountFig \
= band_unfold.plot_ebs(save_figure_dir=save_to_dir, save_file_name=save_file_name, CountFig=None, 
                       Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                       mode="fatband", special_kpoints=special_kpoints_pos_labels, 
                       plotSC=True, fatfactor=20, nE=100,smear=0.2, marker='o',
                       threshold_weight=0.01, show_legend=True, 
                       color='gray', color_map='viridis')

# Option 2: Using BandUPpy Plotting module.
# --------------------- Initiate Plotting method ----------------------------
plot_unfold = banduppy.Plotting(save_figure_dir=save_to_dir)

# -------- Read the saved unfolded bandstructure saved data file ------------
unfolded_bandstructure_ = np.loadtxt(f'{save_to_dir}/bandstructure_unfolded.dat')
kpline = np.loadtxt(f'{save_to_dir}/kpoints_unfolded.dat')
with open(f'{save_to_dir}/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels = pickle.load(handle)
    
fig, ax, CountFig \
= plot_unfold.plot_ebs(unfolded_kpoints=f'{save_to_dir}/kpoints_unfolded.dat', 
                       unfolded_bandstructure=f'{save_to_dir}/bandstructure_unfolded.dat', 
                       #unfolded_kpoints=kpline, 
                       #unfolded_bandstructure=unfolded_bandstructure_, 
                       save_file_name=save_file_name, CountFig=None, 
                       Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                       mode="fatband", special_kpoints=special_kpoints_pos_labels, 
                       plotSC=True, fatfactor=20, nE=100,smear=0.2, marker='o',
                       threshold_weight=0.01, show_legend=True, 
                       color='gray', color_map='viridis')

#%% ----------------- Plot the band centers -----------------------------------
plot_unfold = banduppy.Plotting(save_figure_dir=save_to_dir)
fig, ax, CountFig \
    = plot_unfold.plot_ebs(unfolded_kpoints=kpline, 
                           unfolded_bandstructure=unfolded_bandstructure_properties, 
                           save_file_name=save_file_name, CountFig=None, threshold_weight=min_dN,
                           Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                           mode="band_centers", special_kpoints=special_kpoints_pos_labels, 
                           marker='x', smear=0.2, plot_colormap_bandcenter=True,
                           color='black', color_map='viridis')


