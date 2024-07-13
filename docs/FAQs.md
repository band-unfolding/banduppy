# Frequently Asked Questions

__Question:__ I am getting `RuntimeError` in one of my calculations using VASP. How can I resolve this?

`RuntimeError: *** error - computed ncnt=18134 != input nplane=18133`

**Answer**: VASP does not write which plane waves it uses to the `WAVECAR` file. Therefore, when `banduppy` (specifically `IrRep`) reads the band structure, it tries to mimic `VASP`'s selection and ordering of plane waves that fall within the cut-off sphere. Occasionally, due to numerical errors, one code might consider a plane wave within the sphere while the other does not. This discrepancy can result in an extra vector, as seen in the above case. To address this, use a small correction coefficient (`_correct_Ecut0`) value. This slightly adjusts the `Ecut` to either exclude or include plane waves near the boundary of the cut-off sphere. In the example case, use small negative correction to exclude plane waves near the boundary of the cut-off sphere. Use small positive correction when `computed ncnt < input nplane`.

`bands = banduppy.BandStructure(code="vasp", spinor=spinorbit, fPOS = "POSCAR", fWAV = "WAVECAR", _correct_Ecut0=-1e-7)`

__Question:__ BandUP code needed both the primitive unit cell and supercell POSCAR files. But in Banduppy it seems  only POSCAR_SC is needed. What do I need to modify if I want to do calculations with a different primitiveunit cell and where does it come? 

__Answer:__ In a broader sense, for unfolding, we need to know the relationship between the supercell and the reference primitive unit cells on which we want to project your supercell eigenstates. Mathematically, they are related by a matrix transformation: A = M.a. Here, the primitive cell lattice vectors (a_i) (i = 1, 2, 3) form the building unit for the supercell vectors ( A_i), and the two sets of basis vectors are related by the transformation matrix M.

Now, we can provide the primitive unit cell POSCAR (for a) and supercell POSCAR (for A), and software internally calculates the relation between them (M). => BandUP
Alternatively, we can provide the supercell POSCAR (A) and the transformation matrix (M) as inputs, and software internally calculates the reference primitive cell (a). => BandUPpy

The information required is essentially the same for BandUP and BandUPpy, only alternative ways.

Regarding the necessary modifications for different reference primitive unit cell calculations: you need to adjust the `super_cell_size` parameter in for example the `run_banduppy_vasp.py`, which is the primitive-to-supercell transformation matrix.
## ============================================================

__If you are not satisfied with the answers, cannot find an answer to your question, or have new suggestions, please feel free to reach out to us. We are committed to providing the best experience for our users and greatly value your feedback.__


__Have fun with BandUPpy!__

__Best wishes,__  
__The BandUPpy Team__
