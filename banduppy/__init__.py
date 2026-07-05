__version__ = "0.3.5"


import warnings

from  irrep.bandstructure import BandStructure as BandStructure_irrep
from .unfolding import Unfolding, Properties, SaveBandStructuredata, Plotting
from .unfolding_high_level_class import HighLevelProperties


class BandStructure(BandStructure_irrep):

    def __init__(self, *args, **kwargs):
        warnings.warn("irrep.bandstructure.BandStructure is deprecated. Please explicitly use irrep.bandstructure.BandStructure instead.", DeprecationWarning)
        import irrep
        from packaging import version
        if version.parse(irrep.__version__) < version.parse("3.0"):
            super().__init__(*args, **kwargs)
        else:
            self.__dict__.update(irrep.parse_bandstructure(*args, **kwargs).__dict__)
            
        

__all__ = ['Unfolding', 'BandStructure', 'Properties', 'HighLevelProperties', 
           'SaveBandStructuredata', 'Plotting']
