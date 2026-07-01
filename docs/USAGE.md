# Package Documentation

## Let's start ...
#### 0.0. Import modules
```
    import numpy as np
    import pickle
    import banduppy
```
#### 0.1. Define variables
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
#### 1.1. Initiate Unfolding method
```
    band_unfold = banduppy.Unfolding(supercell=super_cell_size,
                                     print_log='high')
```
#### 1.2. Propose degree of unfolding
```
    propose_folding_results = \
    band_unfold.propose_maximum_minimum_folding(PC_BZ_path, min_num_pts=10, max_num_pts=50,
                                                serach_mode='brute_force', draw_plots=True, 
                                                save_plot=False, save_dir='.', 
                                                save_file_name=None)
```
### 2. Create SC kpoints from PC band path
#### 2.1. Creating SC folded kpoints from PC band path
__Note:__ band_unfold.generate_SC_Kpts_from_pc_kpts() can be used to generate SC Kpoints from PC kpoints list.
```
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
```
### 3. Unfold band structure
```
    read_dir = '<path where the vasp output files are>'
```
#### 3.1. Read wave function file
```
    bands = banduppy.BandStructure(code="vasp", spinor=False,
                                   fPOS = f"{read_dir}/POSCAR",
                                   fWAV = f"{read_dir}/WAVECAR")
```
#### 3.2. Unfold the band structures
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
Band ceneters are determined using the SCF algorithm of automatic band center determination from the following papers:
4.1 Medeiros2014: [Paulo V. C. Medeiros, Sven Stafström, and Jonas Björk, Phys. Rev. B **89**, 041407(R) (2014)](http://doi.org/10.1103/PhysRevB.89.041407)
4.2 Mondal2025: TBA
```.
    # -------------------- Initiate Properties method -----------------------------
    unfolded_band_properties = banduppy.Properties(print_log='high')
    #===================================
    bandcenter_algorithm_ = 'Mondal2025' # band center algorithm to use
    gaussian_decay_sigma = 1 # Standard deviation of Gaussian decay in Mondal2025 algo
    min_sum_dNs_for_a_band = 0.1 # Cut off criteria for minimum weights that a band center should have.
    min_dN = 1e-2 # get rid of small weights bands
    #===================================
    threshold_dN_2b_trial_band_center = None # initial guess of the band centers based on the threshold wights.
    err_tolerance = 1e-8 # The tolerance to group the bands set per unique kpoints value.
    prec_pos_band_centers = 1e-5 # in eV # Precision when compared band centers from previous and current SCF
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
```
### 5. Determine effective mass (parabolic and non-parabolic) from part of the band structure or band center data
```
    m_star, optimized_parameters, convergence_measure, one_std_error_popt = \
    unfolded_band_properties.calculate_effecfive_mass(kpath, band_energy,
                                                      initial_guess_params=None,
                                                      ignore_kshift_cbm_fit=True,
                                                      params_bounds = (-np.inf, np.inf),
                                                      fit_weights=None, absolute_weights=False,
                                                      parabolic_dispersion=True,
                                                      hyperbolic_dispersion_positive=False,
                                                      hyperbolic_dispersion_negative=False,
                                                      params_name = ['alpha', 'kshift', 'cbm', 'gamma'])
    #===================================
    band_energy_fit = unfolded_band_properties.effective_mass_fit_functions(kpath, optimized_parameters,
                                                                            parabolic_dispersion=True,
                                                                            hyperbolic_dispersion_positive=False,
                                                                            hyperbolic_dispersion_negative=False)
```
### 6. Determine alloy-scattering potential from part of the band structure or band center data
This is based on the [Pant et. al., APL, 117, 242105 (2020)](http://doi.org/10.1063/5.0027802) paper.
```
    m_star = 0.16 # in m0 unit
    composition = 0.1
    unit_cell_volm = 49.00 # in Angstrom^3
    non_parabolocity_param = 0.6 # eV^-1 <= E(1+non_parabolocity_param*E) = hbar^2 k^2 /2m*
    hbar_ev_s = 6.582119569509067e-16 # eV.s
    U0, pconv = unfolded_band_properties.calculate_alloy_scattering_potential(m_star, unit_cell_volm, composition,
                                                                              band_energy_unfolded_bandstructure,
                                                                              band_width_unfolded_bandstructure,
                                                                              intial_guess_u0=1, fitting_bounds_u0=(0,2),
                                                                              non_parabolocity_param_m_star=non_parabolocity_param)
    print(f'Al content = {composition:.2f}; U0 = {U0[0]:.2f} +- {U0[1]:.2f} eV')

    fit_x = np.linspace(0, max(band_energy_unfolded_bandstructure),20) # in eV
    fit_y = unfolded_band_properties.alloy_scattering_lifetime_function(fit_x, m_star, unit_cell_volm, composition, U0[0],
                                                                        non_parabolocity_param_m_star=non_parabolocity_param) # in eV
```
__Note:__ unfolded_band_properties.alloy_scattering_lifetime_function() returns scattering lifetime in hbar unit, i.e., it returns hbar/tau. To plot 1/tau divide the return from alloy_scattering_lifetime_function() by hbar. hbar should be in eV.s unit. [hbar_ev_s = 6.582119569509067e-16 #eV.s]

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
#### 8.1. Plot band structure
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
#### 8.2. Plot band structure
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
#### 8.3. Plot and overlay multiple band structures
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

#### 8.4. Plot the band centers
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
#### 8.5. Plot the band centers SCF cycles
```
    plot_unfold.plot_scf(kpath_in_angs=kpline, unfolded_bandstructure=unfolded_bandstructure_,
                         al_scf_data=all_scf_data, plot_max_scf_steps=3, save_file_name=save_file_name,
                         Ef=Efermi, Emin=Emin, Emax=Emax, pad_energy_scale=0.5, threshold_weight=min_dN,
                         special_kpoints=special_kpoints_pos_labels, plot_sc_unfold=True, marker='o', 
                         fatfactor=20, smear=0.05, color=None, color_map='viridis', show_legend=False, 
                         plot_colormap_bandcenter=True, show_colorbar=True, colorbar_label=None, 
                         vmin=None, vmax=None, dpi=72)
```
__Note__: One can save the generated figure within the function call `plot_ebs()` and `plot_scf()`. `save_plot_figure()` function can also be used to save a figure instance independently.

### 9. High level function that uses Properties class to do high-level post-processing

```
    prop_cal_hl = banduppy.HighLevelProperties(print_log=None)

```
#### 9.1. Get both the parabolic and hyperbolic effective masses in a single shot
```
    m_star_p_results, m_star_h_results =\
    prop_cal_hl.calculate_effective_masses(band_centers_data_process,
                                           opt_params[ii][0],
                                           opt_params[ii][1],
                                           ignore_kshift_cbm_fit=True,
                                           absolute_weights=False,
                                           hyperbolic_dispersion_positive=True)

    m_star_pa, popt_pa, pcov_pa, params_error_pa = m_star_p_results
    m_star_ha, popt_ha, pcov_ha, params_error_ha = m_star_h_results
```
#### 9.2. Automatically discard outlier band centers for fitting
The algorithm is based on the TBA paper. Refer to the figures (2nd row, 2nd-3rd column) in the README.
```
    # This function applyes SCF algorithm to refine which band centers to fit. So-called
    # bad/outlier band centers will be discarded during fitting.
    m_star_theor = 0.3 # Theoretical m* as the initial guess
    inital_guess_param = [3.81/m_star_theor, 0.1] # [alpha, gamma] 
    band_center_indices_that_used_in_final_fitting, m_star_p_results, m_star_h_results = \
       prop_cal_hl.scf_refine_effecfive_mass_calculator(all_band_centers, 
                                                        inital_guess_param,
                                                        residual_cutoff_in_eV=0.5,
                                                        use_residual_std_dev=False,
                                                        residual_std_dev_start=1.0,
                                                        initial_guess_fit_dispersion=
                                                        'positive_hyperbolic_dispersion',
                                                        ignore_kshift_cbm_fit=True,
                                                        kshift=None, absolute_weights=False,
                                                        max_scf_loop=100)
```
<!-- =========================================================== -->


##
__If you have new suggestions, please feel free to reach out to us. We are committed to providing the best experience for our users and greatly value your feedback.__

__Have fun with BandUPpy!__

__Best wishes,__  
__The BandUPpy Team__
