# ---------------------------- Import modules ---------------------------------
import shutil, os
from subprocess import run
import numpy as np
import pickle
import banduppy

print(f'- BandUPpy version: {banduppy.__version__}')

#%% ----------------------------- Set job -------------------------------------
do_generate_SC_kpts = True
do_self_consistent = False
do_non_self_consistent = False
do_unfold = True
do_plot = True

SimSystem = 'Si'
SimulationParentFolder = '/local/GitHub/TestBandUPpy/QuantumEspresso/' + SimSystem

#%% -------- Define first-principles code executatble path --------------------
qe_bin_path ='pw.x' #'/software/testing/espresso/5.0.3/build01/bin/pw.x'
nproc = 44
npb = 4
qe_exe = f"mpirun -np {nproc} {qe_bin_path} -nk {nproc/npb} -nb {npb}".split()

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
save_to_dir = f'{SimulationParentFolder}/input'
# Wavefunction file path
sim_folder = f'{SimulationParentFolder}/reference_without_SOC' # '<path where the vasp output files are>'
# File format of kpoints file that will be created and saved
kpts_file_format = 'qe' # This will generate qe KPOINTS file format
# QE file prefix
pw_file = f'bulk_{SimSystem}'
# Unfolding results directory
results_dir = f'{SimulationParentFolder}/results'

#%% -------------------- Initiate Unfolding method ----------------------------
if do_generate_SC_kpts:
    print (f"{'='*72}\n- Generating SC Kpoints...")
    band_unfold = banduppy.Unfolding(supercell=super_cell_size, print_log='high')
    
    #%% ------------ Creating SC folded kpoints from PC band path -------------
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
    print ("- Generating SC Kpoints - done")
    
#%% -------------------- Run SCF calculation ----------------------------------
#os.chdir(os.path.dirname(sim_folder))
if do_self_consistent:
    print (f"{'='*72}\n- Self-consistent QE run...")
    # Reading atom positions file
    with open(f"{save_to_dir}/{SimSystem}_SC_positions.dat", 'r') as f:
        read_alat = f.read()
    # Reading kpoints file
    with open(f"{save_to_dir}/K_POINTS_scf.dat", 'r') as f:
        read_kpts = f.read()
    # Reading input file
    with open(f"{save_to_dir}/scf.in", 'r') as f:
        read_scf_in = f.read()
    # Writing input file appended with kpoints
    with open(f"{sim_folder}/{pw_file}_scf.in", 'w') as f:
        f.write(read_scf_in+read_alat+read_kpts)
    # cp pseudo potential files
    shutil.copytree(f"{save_to_dir}/pseudo", f'{sim_folder}/pseudo', dirs_exist_ok=False)
    
    scf_run = run(qe_exe+['-input', f'{pw_file}_scf.in'],
                  stdout=open(f"{sim_folder}/{pw_file}_scf.out","w"))
    
    # backup .save file files
    shutil.copytree(f"{sim_folder}/{pw_file}.save", f'{sim_folder}/{pw_file}_scf.save', dirs_exist_ok=True)

    print ("- Self-consistent QE run - done")

#%% -------------------- Run non-SCF calculation ------------------------------
if do_non_self_consistent:
    print (f"{'='*72}\n- Non-self-consistent QE run...")
    # Reading atom positions file
    with open(f"{save_to_dir}/{SimSystem}_SC_positions.dat", 'r') as f:
        read_alat = f.read()
    # Reading kpoints file
    with open(f"{save_to_dir}/K_POINTS_SC.dat", 'r') as f:
        read_kpts = ''.join([l for l in f.readlines() if not l.lstrip().startswith('!')])
    # Reading input file
    with open(f"{save_to_dir}/nscf.in", 'r') as f:
        read_scf_in = f.read()
    # Writing input file appended with SC banduppy generated kpoints
    with open(f"{sim_folder}/{pw_file}_nscf.in", 'w') as f:
        f.write(read_scf_in+read_alat+read_kpts)

    bands_run = run(qe_exe+['-input', f'{pw_file}_nscf.in'],
                    stdout=open(f"{sim_folder}/{pw_file}_nscf.out","w"))
    
    print ("- Non-self-consistent QE run - done")

#%% ----------------- Unfold the band structures ------------------------------
if do_unfold:
    ## NOTE: For banduppy/irrep ONLY the {pw_file}.save file is needed.
    ## In this example {pw_file}.save == bulk_Si.save folder
    ## *.wfc{N} are temporary file for restart crashed calculation and 
    ## banduppy/irrep woun't read them at all. 
    print (f"{'='*72}\n- Unfolding band structure...")
    unfolded_bandstructure_, kpline \
    = band_unfold.Unfold(ab_initio_code='qe', 
                         unfold_kpts_in_batch=False, 
                         #unfold_kpts_in_batch=True, kpt_batch_size=10, # Memory efficient routine
                         qe_keywards = {'output_file_dir' : sim_folder, 
                                        'save_file_prefix' : pw_file, 
                                        'unfold_spin_channel': None # ['up', 'dw']
                                        },
                         kline_discontinuity_threshold = 0.1, 
                         save_unfolded_kpts = {'save2file': True, 
                                              'fdir': results_dir,
                                              'fname': 'kpoints_unfolded',
                                              'fname_suffix': ''},
                         save_unfolded_bandstr = {'save2file': True, 
                                                'fdir': results_dir,
                                                'fname': 'bandstructure_unfolded',
                                                'fname_suffix': ''})
    print ("- Unfolding - done")

#%% --------------------- Plot band structure ---------------------------------
if do_plot:
    print (f"{'='*72}\n- Plotting band structure...")
    # Fermi energy
    Efermi = None # Energy scale is already scalled during Unfolding 
    # Minima in Energy axis to plot
    Emin = -5
    # Maxima in Energy axis to plot
    Emax = 5
    # Filename to save the figure. If None, figure will not be saved
    save_file_name = 'unfolded_bandstructure.png'
    # Special k-points info
    with open(f'{save_to_dir}/K_POINTS_SpecialKpoints.pkl', 'rb') as handle:
        special_kpts = pickle.load(handle)
    # Plotting
    fig, ax, CountFig \
    = band_unfold.plot_ebs(save_figure_dir=results_dir,
                           save_file_name=save_file_name, CountFig=None, 
                           Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                           mode="fatband", special_kpoints=special_kpts, 
                           plotSC=True, fatfactor=20, nE=100,smear=0.2, 
                           color='red', color_map='viridis', show_colorbar=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting can be done later/separately using Plotting class
    save_file_name = 'unfolded_bandstructure_.png'
    plot_unfold = banduppy.Plotting(save_figure_dir=results_dir)
    fig, ax, CountFig \
    = plot_unfold.plot_ebs(#unfolded_kpoints =f'{results_dir}/kpoints_unfolded.dat', 
                           #unfolded_bandstructure =f'{results_dir}/bandstructure_unfolded.dat', 
                           unfolded_kpoints=kpline,
                           unfolded_bandstructure=unfolded_bandstructure_,
                           save_file_name=save_file_name, CountFig=None,
                           Emin=-5, Emax=+5, pad_energy_scale=0.5,
                           mode="density", special_kpoints=special_kpts,
                           plotSC=True, fatfactor=20, nE=100,smear=0.2,
                           color='red', color_map='viridis', show_colorbar=False)

    print ("- Plotting band structure - done")

