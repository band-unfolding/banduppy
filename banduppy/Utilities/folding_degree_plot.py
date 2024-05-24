import numpy as np
import matplotlib.pyplot as plt
from ..BasicFunctions.general_plot_functions import GeneratePlots

### ===========================================================================

class FoldingDegreePlot(GeneratePlots):
    """
    Plotting number of kpoints in k-path vs degree of folding.

    """
    def __init__(self, fold_results_dictionary, save_figure_dir='.'):
        """
        Initialize the FoldingDegreePlot plotting class.

        Parameters
        ----------
        fold_results_dictionary : dictionary
            Keys are the index of path segment searched from the pathPBZ list supplied.
            Values are 2d array with each row containing number of division in the 1st
            column and percent of folding in the 2nd column.
        save_figure_dir : str/path, optional
            Directory where to save the figure. The default is current directory.

        """
        GeneratePlots.__init__(self, save_figure_dir=save_figure_dir)
        self.proposed_folding_results_ = fold_results_dictionary.copy()
            
    def plot_folding(self, save_file_name=None, CountFig=None, yaxis_label:str='Folding degree (%)',
                     xaxis_label:str='number of kpoints', line_color='k'):
        
        print('- Plotting folding degree...')
        for keys, vals in self.proposed_folding_results_.items():
            fig, ax = plt.subplots()
            ax.set_xlabel(xaxis_label)
            ax.set_ylabel(yaxis_label)
            
            path_start, path_end = vals[0]
            ax.set_title(f'{path_start} --> {path_end}')
            
            XX, YY = vals[1][:,0], vals[1][:,1]
            
            ax.plot(XX, YY, 'o-', color=line_color)
        print('- Done...')
        if save_file_name is None:
            plt.show()
        else:
            CountFig = self.save_figure(save_file_name, CountFig=CountFig)
            plt.close()
        return fig, ax, CountFig