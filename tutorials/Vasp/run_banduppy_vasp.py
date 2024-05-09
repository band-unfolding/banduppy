# ---------------------------- Import modules ---------------------------------
import shutil, os
from subprocess import run
import numpy as np
import pickle
import banduppy

print(f'- Bandup version: {banduppy.__version__}')

#%% ----------------------------- Set job -------------------------------------
spinorbit = True
do_generate_SC_kpts = True
do_self_consistent = True
do_non_self_consistent = True
read_bandstructure = True
do_unfold = True
do_plot = True

SimulationParentFolder = '/local/MyGitHub/TestInstall/Test/Vasp' 

#%% -------- Define first-principles code executatble path --------------------
#vasp_bin_path = "/app/theorie/vasp/bin/vasp_ncl"
vasp_bin_path = "vasp_std"
nproc = 44
vasp = f"mpirun -np {nproc} {vasp_bin_path}".split()

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
save_to_dir = f'{SimulationParentFolder}/input'
# Wavefunction file path
sim_folder = f'{SimulationParentFolder}/reference_without_SOC' # '<path where the vasp output files are>'
# File format of kpoints file that will be created and saved
kpts_file_format = 'vasp' # This will generate vasp KPOINTS file format
# Unfolding results directory
results_dir = f'{SimulationParentFolder}/results'

#%% -------------------- Initiate Unfolding method ----------------------------
if do_generate_SC_kpts:
    print (f"{'='*72}\n- Generating SC Kpoints...")
    band_unfold = banduppy.Unfolding(supercell=super_cell_size, 
                                      print_log='high')
    
    #%% ------------ Creating SC folded kpoints from PC band path -------------
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
    print ("- Generating SC Kpoints - done")
    
#%% -------------------- Run SCF calculation ----------------------------------
os.chdir(os.path.dirname(sim_folder))
if do_self_consistent:
    print (f"{'='*72}\n- Self-consistent VASP run...")
    shutil.copy(f"{save_to_dir}/POSCAR_Si_SC",f"{sim_folder}/POSCAR")
    shutil.copy(f"{save_to_dir}/POTCAR",f"{sim_folder}/POTCAR")
    shutil.copy(f"{save_to_dir}/KPOINTS_scf",f"{sim_folder}/KPOINTS")
    with open(f"{sim_folder}/INCAR","w") as f:
        f.write(open(f"{save_to_dir}/INCAR_scf","r").read().format(LSORBIT = "T" if spinorbit else "F"))
    scf_run = run(vasp,stdout=open(f"{sim_folder}/vasp_scf.out","w"))
    shutil.copy(f"{sim_folder}/OUTCAR",f"{sim_folder}/OUTCAR_scf")
    print ("- Self-consistent VASP run - done")

#%% -------------------- Run non-SCF calculation ------------------------------
if do_non_self_consistent:
    print (f"{'='*72}\n- Non-self-consistent VASP run...")
    shutil.copy(f"{save_to_dir}/KPOINTS_SC",f"{sim_folder}/KPOINTS")
    with open(f"{sim_folder}/INCAR","w") as f:
        f.write(open(f"{save_to_dir}/INCAR_nscf","r").read().format(LSORBIT = "T" if spinorbit else "F", 
                                                                    NBANDS = 96 if spinorbit else 48))
    bands_run = run(vasp,stdout=open(f"{sim_folder}/vasp_nscf.out","w"))
    shutil.copy(f"{sim_folder}/OUTCAR",f"{sim_folder}/OUTCAR_nscf")
    print ("- Non-self-consistent VASP run - done")

#%% ------------------------ Read wave function file --------------------------
if read_bandstructure:
    print (f"{'='*72}\n- Reading bandstructure...")
    bands = banduppy.BandStructure(code="vasp", spinor=False, 
                                   fPOS = f"{sim_folder}/POSCAR",
                                   fWAV = f"{sim_folder}/WAVECAR")
    pickle.dump(bands,open(f"{results_dir}/bandstructure.pickle","wb"))
    print ("- Reading bandstructure - done")
else:
    print (f"{'='*72}\n- Unpickling bandstructure...")
    bands = pickle.load(open(f"{results_dir}/bandstructure.pickle","rb"))  
    print ("- Unpickling - done")

#%% ----------------- Unfold the band structures ------------------------------
if do_unfold:
    print (f"{'='*72}\n- Unfolding band structure...")
    unfolded_bandstructure_, kpline \
    = band_unfold.Unfold(bands, kline_discontinuity_threshold = 0.1, 
                         save_unfolded_kpts = {'save2file': True, 
                                              'fdir': results_dir,
                                              'fname': 'kpoints_unfolded',
                                              'fname_suffix': ''},
                         save_unfolded_bandstr = {'save2file': True, 
                                                'fdir': results_dir,
                                                'fname': 'bandstructure_unfolded',
                                                'fname_suffix': ''})
    print ("- Unfolding - done")
else:
    print (f"{'='*72}\n- Reading bandstructure data from saved file...")
    unfolded_bandstructure_ = np.loadtxt(f'{results_dir}/bandstructure_unfolded.dat')
    kpline = np.loadtxt(f'{results_dir}/kpoints_unfolded.dat')[:,1]
    with open(f'{save_to_dir}/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
        special_kpoints_pos_labels = pickle.load(handle)
    print ("- Reading band structure file - done")

#%% --------------------- Plot band structure ---------------------------------
if do_plot:
    print (f"{'='*72}\n- Plotting band structure...")
    # Fermi energy
    Efermi = 5.9786
    # Minima in Energy axis to plot
    Emin = -5
    # Maxima in Energy axis to plot
    Emax = 5
    # Filename to save the figure. If None, figure will not be saved
    save_file_name = 'unfolded_bandstructure.png'
    
    fig, ax, CountFig \
    = band_unfold.plot_ebs(kpath_in_angs=kpline, 
                            unfolded_bandstructure=unfolded_bandstructure_, 
                            save_figure_dir=results_dir, 
                            save_file_name=save_file_name, CountFig=None, 
                            Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                            mode="density", special_kpoints=special_kpoints_pos_labels, 
                            plotSC=True, fatfactor=20, nE=100,smear=0.2, 
                            scatter_color='red', color_map='viridis')
    print ("- Plotting band structure - done")
