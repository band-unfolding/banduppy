# Frequently Asked Questions

## ===============================================================================
__Problem:__ I am getting `RuntimeError` in one of my calculations using VASP. How can I resolve this?

`RuntimeError: *** error - computed ncnt=18134 != input nplane=18133`

**Answer**: VASP does not write which plane waves it uses to the `WAVECAR` file. Therefore, when `banduppy` (specifically `IrRep`) reads the band structure, it tries to mimic `VASP`'s selection and ordering of plane waves that fall within the cut-off sphere. Occasionally, due to numerical errors, one code might consider a plane wave within the sphere while the other does not. This discrepancy can result in an extra vector, as seen in the above case. To address this, use a small correction coefficient (`_correct_Ecut0`) value. This slightly adjusts the `Ecut` to either exclude or include plane waves near the boundary of the cut-off sphere. In the example case, use small negative correction to exclude plane waves near the boundary of the cut-off sphere. Use small positive correction when `computed ncnt < input nplane`.

`bands = banduppy.BandStructure(code="vasp", spinor=spinorbit, fPOS = "POSCAR", fWAV = "WAVECAR", _correct_Ecut0=-1e-7)`

## ===============================================================================
__Have fun with BandUPpy!__

__Best wishes,__  
__The BandUPpy Team__