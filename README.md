# BandUP: Band Unfolding code for Plane-wave based calculations             
# BandUPpy - Python version of BandUP (not to be confused with bandupy - the interface and plotting tool of BandUP)

This is a python version to the BandUP code, made in order to restore 
support of the modern versions of QuantumEspresso and other codes. In order ot read the wavefunctions
stored by abiinitio codes, the routines of irrep [irrep](https://github.com/stepan-tsirkin/irrep) are used. 


If you find BandUPpy useful, but missing any important part of the functionality of the original bandUP code,
please let me know.

Author of BandUPpy : Stepan S. Tsirkin, University of Zurich, stepan.tsirkin@uzh.ch

Author of BandUP : Paulo V. C. Medeiros, Linköping University, (at present: SMHI, the Swedish Meteorological and Hydrological Institute)

### please refer to 
##### <http://www.ifm.liu.se/theomod/compphys/band-unfolding>
##### <https://github.com/band-unfolding/bandup>

BandUPpy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BandUPpy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BandUP.  If not, see <http://www.gnu.org/licenses/>.


<!-- =========================================================== -->
##  Plane-wave codes currently supported by BandUPpy

At the moment, BandUP can parse wavefunctions generated by: 

  * [VASP](http://www.vasp.at)
  * [Quantum ESPRESSO](http://www.quantum-espresso.org)
  * [ABINIT](http://www.abinit.org)
  * any code that has interface to [Wannier90](http://wannier.org) (via reading the UNK* and *.eig files)

<!-- =========================================================== -->


<!-- =========================================================== -->
##  Plane-wave codes currently supported by original BandUP: 

At the moment, BandUP can parse wavefunctions generated by: 

  * [VASP](http://www.vasp.at)
  * [Quantum ESPRESSO](http://www.quantum-espresso.org) (issues arise with new versions)
  * [ABINIT](http://www.abinit.org)
  * [CASTEP](http://www.castep.org)  (use the main code of BandUP, not the bandupy)
    (tested with the academic version; currently only 
     available on request)  

<!-- =========================================================== -->


## Publications:

If you use BandUPpy  in
your work, you should:

  * **State EXPLICITLY that you've used the BandUP code** 
    (or a modified version of it, if this is the case).
  * **Read and cite the following papers** (and the appropriate
    references therein):
    
>> Paulo V. C. Medeiros, Sven Stafström, and Jonas Björk,
   [Phys. Rev. B **89**, 041407(R) (2014)](
    http://dx.doi.org/10.1103/PhysRevB.89.041407)  
>> Paulo V. C. Medeiros, Stepan S. Tsirkin, Sven Stafström, and Jonas Björk,
   [Phys. Rev. B **91**, 041116(R) (2015)](
    http://dx.doi.org/10.1103/PhysRevB.91.041116)


if you use BandUPpy,  please also cite

>> Mikel Iraola, Juan L. Mañes, Barry Bradlyn, Titus Neupert, Maia G. Vergniory, Stepan S. Tsirkin 
   "IrRep: symmetry eigenvalues and irreducible representations of ab initio band structures", [arXiv:2009.01764](https://arxiv.org/abs/2009.01764)

An appropriate way of acknowledging the use of BandUP in your
publications would be, for instance, adding a sentence like: 

         "The unfolding has been performed using the BandUP code"

followed by the citation to our papers.

