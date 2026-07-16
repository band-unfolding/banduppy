import numpy as np
import matplotlib.pyplot as plt
from ..BasicFunctions.general_plot_functions import _GeneratePlots

### ===========================================================================

class _FoldingDegreePlot(_GeneratePlots):
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
        _GeneratePlots.__init__(self, save_figure_dir=save_figure_dir)
        self.proposed_folding_results_ = fold_results_dictionary.copy()
            
    def _plot_folding(self, save_file_name=None, CountFig=None, 
                      left_yaxis_label:str='Number of SC Kpoints',
                      right_yaxis_label:str='Folding degree (%)',
                      xaxis_label:str='Number of PC kpoints', 
                      line_color=('k','r'), show_plot:bool=True):
        
        print('- Plotting folding degree...')
        for keys, vals in self.proposed_folding_results_.items():
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            
            ax.set_xlabel(xaxis_label)
            ax.set_ylabel(left_yaxis_label, color=line_color[0])
            ax2.set_ylabel(right_yaxis_label, color=line_color[1])
            
            path_start, path_end = vals[0]
            ax.set_title(f'{path_start} --> {path_end}')
            
            XX, YY_L, YY_R = vals[1][:,0], vals[1][:,1], vals[1][:,-1]
            
            ax.plot(XX, YY_L, 'o-', color=line_color[0])
            ax2.plot(XX, YY_R, 's-', color=line_color[1])
            
            xpad = abs(XX[-1] - XX[0])*(0.02)
            xmin, xmax = XX[0]-xpad, XX[-1]+xpad
            ax.set_xlim(xmin=xmin, xmax=xmax)
            ax.set_ylim(xmin, xmax)
            ax.plot([xmin, xmax],[xmin, xmax], c='gray', ls='--')
            ax2.set_ylim(-2,102)
            
        print('- Done')
        if save_file_name is None:
            if show_plot: 
                plt.show()
            else:
                plt.close()
        else:
            CountFig = self._save_figure(save_file_name, CountFig=CountFig)
            plt.close()
        return fig, ax, CountFig