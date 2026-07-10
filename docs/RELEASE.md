__Latest release: v0.3.6__

__v0.3.6__

* `irrep>=3.0` compatibility
* `banduppy.Unfold()` now support unfolding in batch (eywords `unfold_kpts_in_batch:bool=False, kpt_batch_size:int=10`). This allows memory efficient unfolding
* WANING msg requesting user to move to batch unfolding routine is printed if wave function file is large (default is >5gb).
* `banduppy.Unfold()` now supports additional keywords that internally take care of bandstructure instance generation
* `banduppy.Unfold()` is now compatible with direct user specified `bandstructure` instance from external packages like irrep. If `bandstructure` is not user supplied, banduppy internally calls irrep.bandstructure to generate `bandstructure` instances.
* For VASP, banduppy can now read `vasprun.xml` file to get Fermi energy info. User specified Efermi is also supported.
* Zero weight k-point method is implemented for VASP.
* testsuite improved
* testsuite for QE added

__v0.3.5__

* Moved setup.py to project.toml
* Added testsuite for VASP
* Merging save_all_kpts and save_sc_kpts to single keyword save_kpts.
* Improved band center determination algorithm
* Added alloy disorder scattering potential calculator
* Documentation updated for return arguments from effective_mass_calculator function
* Added high-level properties calculator
* Added color and marker arguments option for supercell bands in plot_ebs()

__v0.3.4__

* Added Quantum Espresso tutorial
* Improving collect_bandstr_data_only_in_energy_window() functionality 
* Upgraded return of band_ceneter_determination() function
* Added effective mass error estimation from fitting error

__v0.3.3__

* Bug fix unfolding kpline in A^-1
* Updated install requirements. Bug found in irrep-1.9.0 and 1.9.1.

__v0.3.2__

* Bug fix for PC band path Cartesian coordinate conversion
* Improved band structures plotting
* Implemented plotting band centers at each SCF cycles
* Implemened save band structure data independently
* Implemented effective mass modules
* Added version track file
* Added bibliography file (.bib) for referencing
* Added tips and tricks in README
* Documentation improved

__v0.3.1__

* Package release in PyPi repository

__v0.2.1__

* Package restructuring with significant code reimplementation
* Documentation improved
* Implemented folding degree module
* Band centers determination algorithm implemented
* Unfolded band structures plotting module improved

__v0.2.0__

* Band unfolding implemented
* Plot unfolded band structures implemented





