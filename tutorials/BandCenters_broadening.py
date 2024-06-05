# ---------------------------- Import modules ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle
import banduppy

print(f'- Bandup version: {banduppy.__version__}')

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
# Experience suggests to tune the following 3 variables for improving band centers determination
min_dN = 1e-5  # 1e-4 should be OK
min_sum_dNs_for_a_band = 0.05 # 1e-1 is OK
# I used to set threshold_dN_2b_trial_band_center = 1e-1 or 2e-1 (work for most of the cases), but this is a complicated parameter...
threshold_dN_2b_trial_band_center = 0.05

# These next two variables do not have strong influence on determining band centers
prec_pos_band_centers = 1e-5 # in eV
err_tolerance = 1e-8
#===================================

unfolded_bandstructure_properties, all_scf_data = \
    unfolded_band_properties.band_centers_broadening_bandstr(unfolded_bandstructure_, 
                                                             min_dN_pre_screening=min_dN,
                                                             threshold_dN_2b_trial_band_center=threshold_dN_2b_trial_band_center,
                                                             min_sum_dNs_for_a_band=min_sum_dNs_for_a_band, 
                                                             precision_pos_band_centers=prec_pos_band_centers,
                                                             err_tolerance_compare_kpts=err_tolerance,
                                                             collect_scf_data=False,
                                                             save_data = {'save2file': True})
#%% ============================== Plottings ==================================
plot_unfold = banduppy.Plotting(save_figure_dir=SimulationParentFolder)

Efermi = 5.5305; Emin = -5; Emax = 5

fig, ax, CountFig \
= plot_unfold.plot_ebs(kpath_in_angs=kpline, 
                        unfolded_bandstructure=unfolded_bandstructure_, 
                        save_file_name=None, CountFig=None, threshold_weight=min_dN,
                        Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                        threshold_weight=min_dN,
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
