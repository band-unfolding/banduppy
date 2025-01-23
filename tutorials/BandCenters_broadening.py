# ---------------------------- Import modules ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle
import banduppy

print(f'- BandUPpy version: {banduppy.__version__}')

#%% ----------------------------- Set job -------------------------------------
SimulationParentFolder = '/local/MyGitHub/TestInstall/Test/OverLayBandStructures/' 

#%% ------------------------ Define variables ---------------------------------
print_output = 'high'
#%% -------------------- Initiate Properties method ---------------------------
unfolded_band_properties = banduppy.Properties(print_log=print_output)

#%%  ---------- Read band structure data file ---------------------------------
print (f"{'='*72}\n- Reading band structure data from saved file...")
unfolded_bandstructure_ = np.loadtxt(f'{SimulationParentFolder}/Si_Ge_supercell/bandstructure_unfolded.dat')
kpline = np.loadtxt(f'{SimulationParentFolder}/Si_Ge_supercell/kpoints_unfolded.dat')[:,1]
with open(f'{SimulationParentFolder}/Si_Ge_supercell/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels = pickle.load(handle)
print ("- Reading band structure file - done")

#%% ------------------- Determine band centers and band widths ----------------
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
#%% ============================== Plottings ==================================
plot_unfold = banduppy.Plotting(save_figure_dir=SimulationParentFolder)

Efermi = 5.5305; Emin = -5; Emax = 5

fig, ax, CountFig \
= plot_unfold.plot_ebs(kpath_in_angs=kpline, 
                        unfolded_bandstructure=unfolded_bandstructure_, 
                        save_file_name=None, CountFig=None, threshold_weight=min_dN,
                        Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                        mode="fatband", special_kpoints=special_kpoints_pos_labels, 
                        plotSC=False, fatfactor=10, nE=100,smear=0.2, show_plot=False,
                        color='k', color_map='viridis', show_legend=False)

fig, ax, CountFig \
= plot_unfold.plot_ebs(ax=ax, kpath_in_angs=kpline, 
                        unfolded_bandstructure=unfolded_bandstructure_properties, 
                        save_file_name=None, CountFig=None, 
                        Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                        mode="band_centers", special_kpoints=special_kpoints_pos_labels, 
                        plotSC=True, fatfactor=20, nE=100,smear=0.2, 
                        color='k', color_map='viridis', show_legend=False,
                        plot_colormap_bandcenter=True)
ax.set_title('black circles: Org. bandstructure, cross: band centers', size=18)
plt.savefig(f'{SimulationParentFolder}/band_center_width', bbox_inches='tight', dpi=300)
