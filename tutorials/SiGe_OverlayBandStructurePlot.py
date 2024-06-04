# ---------------------------- Import modules ---------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pickle
import banduppy

print(f'- Bandup version: {banduppy.__version__}')
print('Note: The lattice parameters for all three Si, Ge and Si-Ge supercells are kept same.')

#%% ----------------------------- Set job -------------------------------------
SimulationParentFolder = '/local/MyGitHub/TestInstall/Test/OverLayBandStructures/' 

#%% ------------------------ Define variables ---------------------------------
# supercell : 4X4X2 supercell == np.diag([4,4,2]) or
super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] 

#%% -------------------- Initiate Plotting method -----------------------------
plot_unfold = banduppy.Plotting(save_figure_dir=SimulationParentFolder)

#%%  
print (f"{'='*72}\n- Reading band structure data from saved file...")
unfolded_bandstructure_1 = np.loadtxt(f'{SimulationParentFolder}/Si_Ge_supercell/bandstructure_unfolded.dat')
kpline1 = np.loadtxt(f'{SimulationParentFolder}/Si_Ge_supercell/kpoints_unfolded.dat')[:,1]
with open(f'{SimulationParentFolder}/Si_Ge_supercell/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels1 = pickle.load(handle)

unfolded_bandstructure_2 = np.loadtxt(f'{SimulationParentFolder}/Si_supercell/bandstructure_unfolded.dat')
kpline2 = np.loadtxt(f'{SimulationParentFolder}/Si_supercell/kpoints_unfolded.dat')[:,1]
with open(f'{SimulationParentFolder}/Si_supercell/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels2 = pickle.load(handle)
    
unfolded_bandstructure_3 = np.loadtxt(f'{SimulationParentFolder}/Ge_supercell/bandstructure_unfolded.dat')
kpline3 = np.loadtxt(f'{SimulationParentFolder}/Ge_supercell/kpoints_unfolded.dat')[:,1]
with open(f'{SimulationParentFolder}/Ge_supercell/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels3 = pickle.load(handle)
print ("- Reading band structure file - done")

#%% --------------------- Plot band structure ---------------------------------
print (f"{'='*72}\n- Plotting band structure...")
# Minima in Energy axis to plot
Emin = -5
# Maxima in Energy axis to plot
Emax = 5
# Filename to save the figure. If None, figure will not be saved
save_file_name = 'SiGeOverlayBandStructure.png'

fig, ax, CountFig \
= plot_unfold.plot_ebs(kpath_in_angs=kpline1, 
                        unfolded_bandstructure=unfolded_bandstructure_1,  
                        save_file_name=None, CountFig=None, threshold_weight=0.1,
                        Ef=5.5305, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                        mode="fatband", special_kpoints=special_kpoints_pos_labels1, 
                        plotSC=False, fatfactor=20, nE=100,smear=0.2, show_plot=False,
                        color='red', color_map='viridis', show_legend=False)

_, ax, CountFig \
= plot_unfold.plot_ebs(ax=ax, kpath_in_angs=kpline2, 
                        unfolded_bandstructure=unfolded_bandstructure_2, 
                        save_file_name=None, CountFig=None, threshold_weight=0.1,
                        Ef=5.9786, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                        mode="fatband", special_kpoints=None, show_plot=False, 
                        plotSC=False, marker='x', fatfactor=10, nE=100,smear=0.2, 
                        color='black', color_map='viridis', show_legend=False)

_, ax, CountFig \
= plot_unfold.plot_ebs(ax=ax, kpath_in_angs=kpline3, 
                        unfolded_bandstructure=unfolded_bandstructure_3, threshold_weight=0.1,
                        save_file_name=None, CountFig=None, 
                        Ef=5.1434, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                        mode="fatband", special_kpoints=None, show_plot=False,
                        plotSC=False, marker='x', fatfactor=10, nE=100,smear=0.2, 
                        color='blue', color_map='viridis', show_legend=False)
ax.set_title('Si-Ge: Red, pure Si: black, pure Ge: blue', size=18)
plt.savefig(f'{SimulationParentFolder}/{save_file_name}', bbox_inches='tight', dpi=300)
print ("- Plotting band structure - done")