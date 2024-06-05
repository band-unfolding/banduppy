# ---------------------------- Import modules ---------------------------------
import numpy as np
import pickle
import banduppy

print(f'- Bandup version: {banduppy.__version__}')

#%% ------------------------ Define variables ---------------------------------
# supercell : 4X4X2 supercell == np.diag([4,4,2]) or
super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] 
# k-path: L-G-X-U,K-G. If the segmant is skipped, put a None between nodes.
PC_BZ_path = [[1/2,1/2,1/2], [0,0,0],[1/2,0,1/2], [5/8,1/4,5/8], None, [3/8,3/8,3/4], [0,0,0]] 
# Number of k-points in each path segments. or, one single number if they are same.
npoints_per_path_seg = (23,27,9,29) 
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
                                              save_all_kpts = save_to_file,
                                              save_sc_kpts = save_to_file,
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
= band_unfold.Unfold(bands, kline_discontinuity_threshold = 0.1, 
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
# = band_unfold.Unfold(bands, PBZ_kpts_list_full=kpointsPBZ_full, 
#                      SBZ_kpts_list=kpointsSBZ, 
#                      SBZ_PBZ_kpts_map=SBZ_PBZ_kpts_mapping,
#                      kline_discontinuity_threshold = 0.1, 
#                      save_unfolded_kpts = {'save2file': True, 
#                                           'fdir': save_to_dir,
#                                           'fname': 'kpoints_unfolded',
#                                           'fname_suffix': ''},
#                      save_unfolded_bandstr = {'save2file': True, 
#                                             'fdir': save_to_dir,
#                                             'fname': 'bandstructure_unfolded',
#                                             'fname_suffix': ''})

#%% ---------------- Determine band centers and band width --------------------
# This uses SCF algorithm of automatic band center determination from 
# PRB 89, 041407(R) (2014) paper.
# -------------------- Initiate Properties method -----------------------------
unfolded_band_properties = banduppy.Properties(print_log='high')

# Experience suggests to tune the following 3 variables for improving band centers determination
min_dN = 1e-5 
min_sum_dNs_for_a_band = 0.05 
threshold_dN_2b_trial_band_center = 0.05

# These next two variables do not have strong influence on determining band centers
prec_pos_band_centers = 1e-5 # in eV
err_tolerance = 1e-8
#===================================

unfolded_bandstructure_properties, all_scf_data = \
    unfolded_band_properties.band_centers_broadening_bandstr(unfolded_bandstructure_, 
                                                             min_dN_pre_screening=min_dN,
                                                             threshold_dN_2b_trial_band_center=
                                                             threshold_dN_2b_trial_band_center,
                                                             min_sum_dNs_for_a_band=min_sum_dNs_for_a_band, 
                                                             precision_pos_band_centers=prec_pos_band_centers,
                                                             err_tolerance_compare_kpts=err_tolerance,
                                                             collect_scf_data=False)

#%% --------------------- Plot band structure ---------------------------------
# Fermi energy
Efermi = 5.9740
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
kpline = np.loadtxt(f'{save_to_dir}/kpoints_unfolded.dat')[:,1]
with open(f'{save_to_dir}/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels = pickle.load(handle)
    
fig, ax, CountFig \
= plot_unfold.plot_ebs(kpath_in_angs=kpline, unfolded_bandstructure=unfolded_bandstructure_, 
                       save_file_name=save_file_name, CountFig=None, 
                       Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                       mode="fatband", special_kpoints=special_kpoints_pos_labels, 
                       plotSC=True, fatfactor=20, nE=100,smear=0.2, marker='o',
                       threshold_weight=0.01, show_legend=True, 
                       color='gray', color_map='viridis')

#%% ----------------- Plot the band centers -----------------------------------
plot_unfold = banduppy.Plotting(save_figure_dir=save_to_dir)
fig, ax, CountFig \
    = plot_unfold.plot_ebs(kpath_in_angs=kpline, 
                           unfolded_bandstructure=unfolded_bandstructure_properties, 
                           save_file_name=save_file_name, CountFig=None, threshold_weight=min_dN,
                           Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                           mode="band_centers", special_kpoints=special_kpoints_pos_labels, 
                           marker='x', smear=0.2, plot_colormap_bandcenter=True,
                           color='black', color_map='viridis')


