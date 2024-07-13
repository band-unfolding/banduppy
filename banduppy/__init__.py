__version__ = "0.3.3"


import irrep
if not ('1.6.2' <= irrep.__version__ <= '1.8.3'):
    raise ImportError("Critical bugs found in irrep-1.6.1 and >1.8.3 versions. We request BandUPpy users to use irrep-1.8.3 until the next release of the irrep package.")

from  irrep.bandstructure import BandStructure


from  irrep.bandstructure import BandStructure

from .unfolding import Unfolding, Properties, SaveBandStructuredata, Plotting

__all__ = ['Unfolding', 'BandStructure', 'Properties', 'SaveBandStructuredata', 'Plotting']
