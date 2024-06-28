__version__ = "0.3.3"

import irrep
if  irrep.__version__ <"1.6.2" :
    raise ImportError("A critical bug was found in irrep-1.6.1, which caused incorrect results for unfolding with spin-orbit. Please update irrep to 1.6.2 or newer.")
if  irrep.__version__ == "1.9.0" :
    raise ImportError("A critical bug was found in irrep-1.9.0 for vasp WAVECAR reading. Please update irrep to other or newer version.")
from  irrep.bandstructure import BandStructure

from .unfolding import Unfolding, Properties, SaveBandStructuredata, Plotting

__all__ = ['Unfolding', 'BandStructure', 'Properties', 'SaveBandStructuredata', 'Plotting']
