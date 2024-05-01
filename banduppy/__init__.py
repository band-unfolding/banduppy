import irrep
if  irrep.__version__ <"1.6.2" :
    raise ImportError("A critical bug was found in irrep-1.6.1, which caused incorrect results for unfolding with spin-orbit. Please ipdate irrep to 1.6.2 or newer (when available)")

from  irrep.bandstructure import BandStructure
from .unfolding import Unfolding
from ._version import _pkg_version

__version__ = _pkg_version
__all__ = ['Unfolding', 'BandStructure']
