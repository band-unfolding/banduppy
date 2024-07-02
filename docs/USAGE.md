# Package Documentation

## ===============================================================================
## Let's start ...
#### ---------------------------- Import modules ---------------------------------
```
    import numpy as np
    import pickle
    import banduppy
```
#### ------------------------ Define variables ---------------------------------
##### Note: Only the VASP KPOINTS file format is implemented so far.
```
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
```
### 1. Estimate the best choice of number of kpoints to use in effective band structure each k-path segments considering maximizing or minimizing folding in the supercell.
#### __Motivation:__ Maximizing folding helps minimizing computational resource. 
__Definition:__ $\text{Folding percent} = \frac{\text{no. of unique PC kpoints} \ -\ \text{no. of folded SC Kpoints}}{\text{no. of unique PC kpoints}}\times100$
#### ------------------ Initiate Unfolding method -----------------------------
```
    band_unfold = banduppy.Unfolding(supercell=super_cell_size,
                                     print_log='high')
```
#### ------------- Propose degree of unfolding --------------------------------
```
    propose_folding_results = \
    band_unfold.propose_maximum_minimum_folding(PC_BZ_path, min_num_pts=10, max_num_pts=50,
                                                serach_mode='brute_force', draw_plots=True, 
                                                save_plot=False, save_dir='.', 
                                                save_file_name=None)
```
### 2. Create SC kpoints from PC band path
#### ------------ Creating SC folded kpoints from PC band path -----------------
__Note:__ band_unfold.generate_SC_Kpts_from_pc_kpts() can be used to generate SC Kpoints from PC kpoints list.
```
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
```
### 3. Unfold band structure
```
    read_dir = '<path where the vasp output files are>'
```
#### ------------------------ Read wave function file --------------------------
```
    bands = banduppy.BandStructure(code="vasp", spinor=False,
                                   fPOS = f"{read_dir}/POSCAR",
                                   fWAV = f"{read_dir}/WAVECAR")
```
#### ----------------- Unfold the band structures ------------------------------
```
    # save2file : Save unfolded kpoints or not? 
    # fdir : Directory path where to save the file.
    # fname : Name of the file.
    # fname_suffix : Suffix to add to the file name.
```
__Option 1:__ Continue with previous instance.
```
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
```
__Option 2:__ If this part is used independently from the above instances re-initiate the Unfolding module.
```
    # --------------------- Initiate Unfolding method --------------------------
    band_unfold = banduppy.Unfolding(supercell=super_cell_size, print_info='high')

    # ----------------- Unfold the band structures ------------------------------
    unfolded_bandstructure_, kpline \
    = band_unfold.Unfold(bands, PBZ_kpts_list_full=kpointsPBZ_full, 
                         SBZ_kpts_list=kpointsSBZ, 
                         SBZ_PBZ_kpts_map=SBZ_PBZ_kpts_mapping,
                         kline_discontinuity_threshold = 0.1, 
                         save_unfolded_kpts = {'save2file': True, 
                                              'fdir': save_to_dir,
                                              'fname': 'kpoints_unfolded',
                                              'fname_suffix': ''},
                         save_unfolded_bandstr = {'save2file': True, 
                                                'fdir': save_to_dir,
                                                'fname': 'bandstructure_unfolded',
                                                'fname_suffix': ''})
```
### 4. Determine band centers and band width
Band ceneters are determined using the SCF algorithm of automatic band center determination from [Paulo V. C. Medeiros, Sven Stafström, and Jonas Björk, Phys. Rev. B **89**, 041407(R) (2014)](http://doi.org/10.1103/PhysRevB.89.041407) paper.
```.
    # -------------------- Initiate Properties method -----------------------------
    unfolded_band_properties = banduppy.Properties(print_log='high')
    #===================================
    min_dN = 1e-5 # get rid of small weights bands
    threshold_dN_2b_trial_band_center = 0.05 # initial guess of the band centers based on the threshold wights.
    min_sum_dNs_for_a_band = 0.05 # Cut off criteria for minimum weights that a band center should have.
    #===================================
    err_tolerance = 1e-8 # The tolerance to group the bands set per unique kpoints value.
    prec_pos_band_centers = 1e-5 # in eV # Precision when compared band centers from previous and current SCF
    #===================================
    unfolded_bandstructure_properties, all_scf_data = \
        unfolded_band_properties.band_centers_broadening_bandstr(unfolded_bandstructure_, 
                                                                 min_dN_pre_screening=min_dN,
                                                                 threshold_dN_2b_trial_band_center=
                                                                 threshold_dN_2b_trial_band_center,
                                                                 min_sum_dNs_for_a_band=min_sum_dNs_for_a_band, 
                                                                 precision_pos_band_centers=prec_pos_band_centers,
                                                                 err_tolerance_compare_kpts_val=err_tolerance,
                                                                 collect_scf_data=False)
```
### 5. Determine effective mass (parabolic and non-parabolic) from part of the band structure or band center data
```
    m_star, optimized_parameters, convergence_measure = \
    unfolded_band_properties.calculate_effecfive_mass(kpath, band_energy,
                                                      initial_guess_params=None,
                                                      params_bounds = (-np.inf, np.inf),
                                                      fit_weights=None, absolute_weights=False,
                                                      parabolic_dispersion=True,
                                                      hyperbolic_dispersion_positive=False,
                                                      hyperbolic_dispersion_negative=False,
                                                      params_name = ['alpha', 'kshift', 'cbm', 'gamma'])
    #===================================
    band_energy_fit = unfolded_band_properties.fit_functions(kpath, optimized_parameters,
                                                            parabolic_dispersion=True,
                                                            hyperbolic_dispersion_positive=False,
                                                            hyperbolic_dispersion_negative=False)
```
### 6. Determine alloy-scattering potential from part of the band structure or band center data
This is based on the ... paper.
```
    TBA
```
### 7. Save unfolded band structure and band center data
One can save the generated data within the function call for unfolding and band ceneter determination routines as shown above. [recommened]

__However,__ you may want save the generated data (from the above function calls) after some post processing, for e.g., you want to save part of the data only. __In such cases,__ you can use functions from `SaveBandStructuredata` class to save those data. Note that the data format should be compatible with the required data format for each functions.

### 8. Plot unfolded band structure (scatter plot/density plot/band_centers plot)
```
    # Fermi energy
    Efermi = 5.9740
    # Minima in Energy axis to plot
    Emin = -5
    # Maxima in Energy axis to plot
    Emax = 5
    # Filename to save the figure. If None, figure will not be saved
    save_file_name = 'unfolded_bandstructure.png'
```
__Option 1:__ Continue with previous instance.
#### --------------------- Plot band structure ---------------------------------
```
    fig, ax, CountFig \
    = band_unfold.plot_ebs(save_figure_dir=save_to_dir, save_file_name=save_file_name, CountFig=None, 
                          Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                          mode="density", special_kpoints=special_kpoints_pos_labels, 
                          plotSC=True, fatfactor=20, nE=100,smear=0.2, marker='o',
                          threshold_weight=0.01, show_legend=True,
                          color='gray', color_map='viridis')
```
__Option 2:__ Using BandUPpy Plotting module.
```
    # --------------------- Initiate Plotting method ----------------------------
    plot_unfold = banduppy.Plotting(save_figure_dir=save_to_dir)
    
    # -------- Read the saved unfolded bandstructure saved data file ------------
    unfolded_bandstructure_ = np.loadtxt(f'{save_to_dir}/bandstructure_unfolded.dat')
    kpline = np.loadtxt(f'{save_to_dir}/kpoints_unfolded.dat')[:,1]
    with open(f'{save_to_dir}/KPOINTS_SpecialKpoints.pkl', 'rb') as handle:
        special_kpoints_pos_labels = pickle.load(handle)
```
#### --------------------- Plot band structure ----------------------------------
```
    
    fig, ax, CountFig \
    = plot_unfold.plot_ebs(kpath_in_angs=kpline, unfolded_bandstructure=unfolded_bandstructure_, 
                           save_file_name=save_file_name, CountFig=None, 
                           Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                           mode="density", special_kpoints=special_kpoints_pos_labels, 
                           plotSC=True, fatfactor=20, nE=100,smear=0.2, marker='o',
                           threshold_weight=0.01, show_legend=True, 
                           color='gray', color_map='viridis')
```
#### --------- Plot and overlay multiple band structures ------------
```
    fig, ax, CountFig \
    = plot_unfold.plot_ebs(kpath_in_angs=kpline1, 
                            unfolded_bandstructure=unfolded_bandstructure_1, 
                            save_file_name=None, CountFig=None, threshold_weight=0.1,
                            Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                            mode="fatband", special_kpoints=special_kpoints_pos_labels1, 
                            plotSC=True, fatfactor=20, nE=100, smear=0.2,
                            color='red', color_map='viridis', show_plot=False)
    
    fig, ax, CountFig \
    = plot_unfold.plot_ebs(ax=ax, kpath_in_angs=kpline1, 
                            unfolded_bandstructure=unfolded_bandstructure_2, 
                            save_file_name=save_file_name, CountFig=None, 
                            Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                            mode="fatband", special_kpoints=None, marker='x',
                            smear=0.2, color='black', color_map='viridis')
```

#### ----------------- Plot the band centers -----------------------------------
```
    fig, ax, CountFig \
        = plot_unfold.plot_ebs(kpath_in_angs=kpline, 
                               unfolded_bandstructure=unfolded_bandstructure_properties, 
                               save_file_name=save_file_name, CountFig=None, threshold_weight=min_dN,
                               Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, 
                               mode="band_centers", special_kpoints=special_kpoints_pos_labels, 
                               marker='x', smear=0.2, plot_colormap_bandcenter=True,
                               color='black', color_map='viridis')
```
#### -------------- Plot the band centers SCF cycles ---------------------
```
    plot_unfold.plot_scf(kpath_in_angs=kpline, unfolded_bandstructure=unfolded_bandstructure_,
                         al_scf_data=all_scf_data, plot_max_scf_steps=3, save_file_name=save_file_name,
                         Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, threshold_weight=min_dN,
                         special_kpoints=special_kpoints_pos_labels, plot_sc_unfold=True, marker='o', 
                         fatfactor=20, smear=0.05, color=None, color_map='viridis', show_legend=False, 
                         plot_colormap_bandcenter=True, show_colorbar=True, colorbar_label=None, 
                         vmin=None, vmax=None, dpi=72)
```
<!-- =========================================================== -->



## ===============================================================================
__Have fun with BandUPpy!__

__Best wishes,__  
__The BandUPpy Team__
