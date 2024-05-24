# ---------------------------- Import modules ---------------------------------
import banduppy as bp
# Note: round() function is used in unfolding_path whenever '%' operation is called
import banduppy_old.unfolding_path as bp_org
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
save_to_file = False 
# Directory to save file
save_to_dir = '<directory to save files>' 
# File format of kpoints file that will be created and saved
kpts_file_format = 'vasp' # This will generate vasp KPOINTS file format

#%% ===========================================================================
unfold_path=bp_org.UnfoldingPath(supercell= super_cell_size, 
                                 pathPBZ=PC_BZ_path,
                                 nk=npoints_per_path_seg, 
                                 labels=special_k_points )  

#%%%
print(unfold_path.kpointsPBZ_index_in_unique)
#%%%
print(unfold_path.kpointsPBZ_unique_index_SBZ)
#%%%
print(unfold_path.kpointsSBZ)

#%% ===========================================================================
band_unfold = bp.Unfolding(supercell=super_cell_size, print_log='high')

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
#%%%
print(band_unfold.kpointsPBZ_index_in_unique)
#%%%
print(band_unfold.kpointsSBZ_index_in_unique)
#%%%
print(band_unfold.SBZ_kpts_list)

#%% ============ Read wave function file ======================================
read_dir = '<path where the vasp output files are>'
bands = bp.BandStructure(code="vasp", spinor=False, 
                         fPOS = f"{read_dir}/POSCAR",
                         fWAV = f"{read_dir}/WAVECAR")

#%% ====================== Unfold band structure ==============================
lll = unfold_path.unfold(bands, break_thresh=0.1,suffix="path")

#%%%
unfolded_bandstructure_, kpline = \
band_unfold.Unfold(bands, kline_discontinuity_threshold = 0.1,
                   save_unfolded_kpts = {'save2file': True, 
                                      'fdir': '.',
                                      'fname': 'kpoints_unfolded',
                                      'fname_suffix': ''}, 
                   save_unfolded_bandstr = {'save2file': True, 
                                        'fdir': '.',
                                        'fname': 'bandstructure_unfolded',
                                        'fname_suffix': ''})

#%% ================ Plot band structure ======================================
#%%%
Efermi = 5.9740
Emin = -5
Emax = 5

# now plot the result as fat band
unfold_path.plot(save_file=None,plotSC=True,Emin=Emin,Emax=Emax,Ef=Efermi,fatfactor=50,mode='fatband') 
# or as a colormap
unfold_path.plot(save_file=None,plotSC=True,Emin=Emin,Emax=Emax,Ef=Efermi,mode='density',smear=0.2,nE=200) 
#%%%
fig, ax, CountFig \
= band_unfold.plot_ebs(save_figure_dir='.', save_file_name=None, CountFig=None, 
                       Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                       mode="density", special_kpoints=special_kpoints_pos_labels, 
                       plotSC=True, fatfactor=50, nE=100,smear=0.2, 
                       scatter_color='red', color_map='viridis')
    
