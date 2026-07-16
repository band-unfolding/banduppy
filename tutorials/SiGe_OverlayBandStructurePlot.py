# ---------------------------- Import modules ---------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pickle
import banduppy

print(f'- BandUPpy version: {banduppy.__version__}')
print('Note: The lattice parameters for all three Si, Ge and Si-Ge supercells are kept same.')

#%% ----------------------------- Set job -------------------------------------
SimulationParentFolder = '/local/GitHub/banduppy/tutorials/OverLayBandStructures/' 

#%% ------------------------ Define variables ---------------------------------
# supercell dimension 
## A general note on supercell dimension: If you have let say a 4X4X2 supercell then you can explicitely write 
## super_cell_size = [[4, 0, 0], [0, 4, 0], [0, 0, 2]] or you can also simply pass
## super_cell_size = np.diag([4,4,2]). Both was defining super_cell_size is allowed.
super_cell_size = [[-1,  1, 1], [1, -1, 1], [1,  1, -1]] # this is for our Si 8-atom conventional unitcell/supercell case. Not to be confused with above note.

#%% -------------------- Initiate Plotting method -----------------------------
plot_unfold = banduppy.Plotting(save_figure_dir=SimulationParentFolder)

#%% --------------------- Plot band structure ---------------------------------
print (f"{'='*72}\n- Plotting band structure...")
# Minima in Energy axis to plot
Emin = -5
# Maxima in Energy axis to plot
Emax = 5
# Filename to save the figure. If None, figure will not be saved
save_file_name = 'SiGeOverlayBandStructure.png'
# Special k-points
with open(f'{SimulationParentFolder}/SiGe/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
    special_kpoints_pos_labels1 = pickle.load(handle)

fig, ax = None, None
color_marker={'SiGe': ('red','o'), 'Si': ('black', 'x'), 'Ge': ('blue', '*')}
for ffolder in ['SiGe', 'Si', 'Ge']:
    fig, ax, CountFig \
    = plot_unfold.plot_ebs(fig=fig, ax=ax, 
                           unfolded_kpoints=f'{SimulationParentFolder}/{ffolder}/kpoints_unfolded.dat', 
                           unfolded_bandstructure=f'{SimulationParentFolder}/{ffolder}/bandstructure_unfolded.dat',  
                           save_file_name=None, CountFig=None, threshold_weight=0.1,
                           Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                           mode="fatband", special_kpoints=special_kpoints_pos_labels1, 
                           plotSC=False, fatfactor=20, nE=100,smear=0.2, append_plts=True,
                           color=color_marker[ffolder][0], marker=color_marker[ffolder][1],
                           color_map='viridis', show_legend=False)
#ax.set_title('Si-Ge: Red, pure Si: black, pure Ge: blue', size=18)
plt.savefig(f'{SimulationParentFolder}/{save_file_name}', bbox_inches='tight', dpi=75)
print ("- Plotting band structure - done")
