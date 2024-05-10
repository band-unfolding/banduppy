import numpy as np
import matplotlib.pyplot as plt
from ..BasicFunctions.general_plot_functions import GeneratePlots
from .EBS_properties import _FormatSpecialKpts

### ===========================================================================

class EBSplot(GeneratePlots, _FormatSpecialKpts):
    """
    Plotting (effective) band structures and related.

    """
    def __init__(self, kpath_in_angs=None, unfolded_bandstructure=None,
                 save_figure_dir='.'):
        """
        Initialize the band structure plotting class.

        Parameters
        ----------
        kpath_in_angs : array, optional
            k on path (in A^-1) coordinate. The default is None.
        unfolded_bandstructure : ndarray, optional
            Unfolded effective band structure data. 
            Format: k on path (A^-1), energy, weight, "Sx, Sy, Sz" if is_spinor.
            The default is None.
        save_figure_dir : str/path, optional
            Directory where to save the figure. The default is current directory.

        """
        GeneratePlots.__init__(self, save_figure_dir=save_figure_dir)
        self.efermi = 0.0
        
        if kpath_in_angs is None:
            try:
                self.kpath_in_angs_ = self.kpline.copy()
            except:
                raise ValueError('No bandstructure data file is found')
        else:
            self.kpath_in_angs_ = kpath_in_angs.copy()
        
        if unfolded_bandstructure is None:
            try:
                self.plot_result = self.unfolded_bandstructure.copy()
            except:
                raise ValueError('No bandstructure data file is found')
        else:
            self.plot_result = unfolded_bandstructure.copy()
        

    # def plot(self, ax=None, save_file_name=None, CountFig=None, Ef=None, Emin=None, 
    #          Emax=None,  pad_energy_scale:float=0.5, threshold_weight:float=None,  
    #          mode:str="fatband", yaxis_label:str='E (eV)', special_kpoints:dict=None, 
    #          plotSC:bool=True, marker='o', fatfactor=20, nE:int=100, smear:float=0.05,  
    #          scatter_color='gray', color_map='viridis', show_legend:bool=True):
    #     """
    #     Scatter/density plot of the band structure.

    #     Parameters
    #     ----------
    #     ax : matplotlib.pyplot axis, optional
    #         Figure axis to plot on. If None, new figure will be created.
    #         The default is None.
    #     save_file_name : str, optional
    #         Name of the figure file. If None, figure will be not saved. 
    #         The default is None.
    #     CountFig: int, optional
    #         Figure count. The default is None.
    #     Ef : float, optional
    #         Fermi energy. If None, set to 0.0. The default is None.
    #     Emin : float, optional
    #         Minimum in energy. The default is None.
    #     Emax : float, optional
    #         Maximum in energy. The default is None.
    #     pad_energy_scale: float, optional
    #         Add padding of pad_energy_scale to minimum and maximum energy if Emin
    #         and Emax are None. The default is 0.5.
    #     threshold_weight : float, optional
    #         The band centers with band weights lower than the threshhold weights 
    #         are discarded. The default is None. If None, this is ignored.
    #     mode : ['fatband','density'], optional
    #         Mode of plot. The default is "fatband".
    #     yaxis_label : str, optional
    #         Y-axis label text. The default is 'E (eV)'.
    #     special_kpoints : dictionary, optional
    #         Dictionary of special kpoints position and labels. If None, ignore
    #         special kpoints. The default is None.
    #     plotSC : bool, optional
    #         Plot supercell bandstructure. The default is True.
    #     marker : matplotlib.pyplot markerMarkerStyle, optional
    #         The marker style. Marker can be either an instance of the class or 
    #         the text shorthand for a particular marker. 
    #         The default is 'o'.
    #     fatfactor : int, optional
    #         Scatter plot marker size. The default is 20.
    #     nE : int, optional
    #         Number of pixels in Energy scale when used 'density' mode. 
    #         The default is 100.
    #     smear : float, optional
    #         Gaussian smearing. The default is 0.05.
    #     scatter_color : str/color, optional
    #         Color of scatter plot of unfolded band structure. The color of supercell
    #         band structures is gray. The default is 'gray'.
    #     color_map: str/ matplotlib colormap
    #         Colormap for density plot. The default is viridis.
    #     show_legend : bool
    #         If show legend or not. The default is True.
        
    #     Raises
    #     ------
    #     ValueError
    #         If plot mode is unknown.

    #     Returns
    #     -------
    #     fig : matplotlib.pyplot.figure
    #         Figure instance. If ax is not None previously generated fig instance
    #         will be used.
    #     ax : Axis instance
    #         Figure axis instance.
    #     CountFig: int or None
    #         Figure count.

    #     """
        
    #     print('- Plotting band structures...')
    #     if ax is None: 
    #         self.fig, ax = plt.subplots()
        
    #     if Ef == 'auto' or Ef is None: 
    #         Ef = self.efermi
    #     # Shift the energy scale to 0 fermi energy level   
    #     if Ef is not None:
    #         self.plot_result[:,2] -= Ef 
    #         ax.axhline(y=0, color='k', ls='--', lw=1)
    #         yaxis_label = r"E$-$E$_\mathrm{F}$ (eV)"
    #         print(f"-- Efermi was set to {Ef} eV")
 
    #     if Emin is None: Emin = self.plot_result[:,2].min() - pad_energy_scale
    #     if Emax is None: Emax = self.plot_result[:,2].max() + pad_energy_scale
        
    #     result_ = self.plot_result[(self.plot_result[:,2] >= Emin - max(smear*10, 0.1)) * 
    #                                (self.plot_result[:,2] <= Emax + max(smear*10, 0.1))]
    #     result = result_[:, 1:]
    #     # Set weights to 0 which are below threshold_weight
    #     if threshold_weight is not None: 
    #         result[result[:, 2] < threshold_weight, 2] = 0
        
    #     # Plot as fat band
    #     if mode == "fatband":
    #         if plotSC:
    #             ax.scatter(result[:, 0], result[:, 1], s=fatfactor, color='gray', label="supercell")
    #         ax.scatter(result[:, 0], result[:, 1], s=result[:, 2]*fatfactor, 
    #                    marker=marker, color=scatter_color, label="unfolded")
    #         if show_legend: ax.legend(loc=1)
    #     elif mode == "density":
    #         energy = np.linspace(Emin, Emax, nE)
    #         density = np.zeros((len(self.kpath_in_angs_),nE), dtype=float)
    #         for k, E, w in result[:, :3]:
    #             ik = np.argmin(abs(k - self.kpath_in_angs_))
    #             # Gaussian smearing
    #             density[ik, :] += w*np.exp(-(energy-E)**2/(2*smear**2)) 
    #         # density=np.log(density)
    #         # density[density<1e-3]=0           
    #         k1 = np.hstack(([self.kpath_in_angs_[0]], 
    #                         (self.kpath_in_angs_[1:]+self.kpath_in_angs_[:-1])/2,
    #                         [self.kpath_in_angs_[-1]]))
    #         E1 = np.hstack(([energy[0]],(energy[1:]+energy[:-1])/2,[energy[-1]]))
    #         # print(k1,E1)
    #         # density=np.pad(density,((0,1),(0,1)))
    #         # print("before",k1.shape,E1.shape,density.shape)
    #         k1, E1 = np.meshgrid(k1,E1)
    #         # print("after",k1.shape,E1.shape,density.shape)
    #         plt.pcolormesh(k1, E1, density.T, cmap=color_map)
    #         plt.colorbar()
    #     else:
    #         raise ValueError("Unknownplot mode: '{}'".format(mode))
            
    #     if special_kpoints is not None:
    #         x_tiks_labels, x_tiks_positions = \
    #             _FormatSpecialKpts._extract_special_kpts_info(special_kpoints, 
    #                                                           self.kpath_in_angs_)
    #         plt.xticks(x_tiks_positions, x_tiks_labels)
    #         # Draw vertical lines
    #         for line_pos in x_tiks_positions:
    #             plt.axvline(x=line_pos, color='k', ls='--', lw=2)
        
    #     ax.set_ylabel(yaxis_label)
    #     ax.set_ylim([Emin, Emax])
    #     ax.set_xlim([self.kpath_in_angs_.min(), self.kpath_in_angs_.max()])

    #     if save_file_name is None:
    #         plt.show()
    #     else:
    #         CountFig = self.save_figure(save_file_name, CountFig=CountFig)
    #         plt.close()
    #     return self.fig, ax, CountFig
    
    def plot(self, ax=None, save_file_name=None, CountFig=None, Ef=None, Emin=None, 
             Emax=None,  pad_energy_scale:float=0.5, threshold_weight:float=None,  
             mode:str="fatband", yaxis_label:str='E (eV)', special_kpoints:dict=None, 
             plotSC:bool=True, marker='o', fatfactor=20, nE:int=100, smear:float=0.05,  
             color='gray', color_map='viridis', plot_colormap_bandcenter:bool=True,
             show_legend:bool=True):
        """
        Scatter/density plot of the band structure.

        Parameters
        ----------
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on. If None, new figure will be created.
            The default is None.
        save_file_name : str, optional
            Name of the figure file. If None, figure will be not saved. 
            The default is None.
        CountFig: int, optional
            Figure count. The default is None.
        Ef : float, optional
            Fermi energy. If None, set to 0.0. The default is None.
        Emin : float, optional
            Minimum in energy. The default is None.
        Emax : float, optional
            Maximum in energy. The default is None.
        pad_energy_scale: float, optional
            Add padding of pad_energy_scale to minimum and maximum energy if Emin
            and Emax are None. The default is 0.5.
        threshold_weight : float, optional
            The band centers with band weights lower than the threshhold weights 
            are discarded. The default is None. If None, this is ignored.
        mode : ['fatband','density', 'band_centers'], optional
            Mode of plot. The default is "fatband".
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plotSC : bool, optional
            Plot supercell bandstructure. The default is True.
        marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style. Marker can be either an instance of the class or 
            the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color of plot of unfolded band structure. The color of supercell
            band structures is gray. The default is 'gray'.
        color_map: str/ matplotlib colormap
            Colormap for density plot. The default is viridis.
        plot_colormap_bandcenter : bool, optional
            If plotting the band ceneters by colormap. The default is True.
        show_legend : bool, optional
            If show legend or not. The default is True.
        
        Raises
        ------
        ValueError
            If plot mode is unknown.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure instance. If ax is not None previously generated fig instance
            will be used.
        ax : Axis instance
            Figure axis instance.
        CountFig: int or None
            Figure count.

        """
        
        print('- Plotting band structures...')
        if ax is None: 
            self.fig, ax = plt.subplots()
        
        if mode != "band_centers" and len(self.plot_result[0]) > 3: 
            self.plot_result = self.plot_result[:, 1:]
        
        if Ef == 'auto' or Ef is None: 
            Ef = self.efermi
        # Shift the energy scale to 0 fermi energy level   
        if Ef is not None:
            self.plot_result[:, 1] -= Ef 
            ax.axhline(y=0, color='k', ls='--', lw=1)
            yaxis_label = r"E$-$E$_\mathrm{F}$ (eV)"
            print(f"-- Efermi was set to {Ef} eV")
 
        if Emin is None: Emin = self.plot_result[:, 1].min() - pad_energy_scale
        if Emax is None: Emax = self.plot_result[:, 1].max() + pad_energy_scale
        
        result = self.plot_result[(self.plot_result[:, 1] >= Emin - max(smear*10, 0.1)) * 
                                  (self.plot_result[:, 1] <= Emax + max(smear*10, 0.1))]

        # Set weights to 0 which are below threshold_weight
        if threshold_weight is not None: 
            result[result[:, -1] < threshold_weight, -1] = 0
        
        # Plot as fat band
        if mode == "fatband":
            ax = self._plot_fatband(result, ax, marker=marker, fatfactor=fatfactor, 
                                    scatter_color=color, show_legend=show_legend,
                                    plotSC=plotSC)
        elif mode == "density":
            ax = self._plot_density(result, ax, Emin, Emax, nE, self.kpath_in_angs_, 
                                    smear, cmap=color_map)
        elif mode == 'band_centers':
            ax = self._plot_band_centers(result, ax, color=color, color_map=color_map,
                                         plot_colormap=plot_colormap_bandcenter)
        else:
            raise ValueError("Unknownplot mode: '{}'".format(mode))
            
        if special_kpoints is not None:
            x_tiks_labels, x_tiks_positions = \
                _FormatSpecialKpts._extract_special_kpts_info(special_kpoints, 
                                                              self.kpath_in_angs_)
            plt.xticks(x_tiks_positions, x_tiks_labels)
            # Draw vertical lines
            for line_pos in x_tiks_positions:
                plt.axvline(x=line_pos, color='k', ls='--', lw=2)
        
        ax.set_ylabel(yaxis_label)
        ax.set_ylim([Emin, Emax])
        ax.set_xlim([self.kpath_in_angs_.min(), self.kpath_in_angs_.max()])

        if save_file_name is None:
            plt.show()
        else:
            CountFig = self.save_figure(save_file_name, CountFig=CountFig)
            plt.close()
        return self.fig, ax, CountFig
    
    @classmethod
    def _plot_fatband(cls, data_4_plot, ax, marker='o', fatfactor=20, scatter_color='gray',
                      cmap='viridis', legend_label='unfolded', show_legend:bool=True,
                      plotSC:bool=True, legend_pos=1):
        """
        Plot fatband scatter plot.

        Parameters
        ----------
        data_4_plot : numpy 2d array
            Data to plot.
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on. 
        marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style. Marker can be either an instance of the class or 
            the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        scatter_color : str/color, optional
            Color of plot of unfolded band structure. The color of supercell
            band structures is gray. The default is 'gray'.
        cmap : str/ matplotlib colormap
            Colormap for density plot. The default is 'viridis'.
        legend_label : str, optional
            Label to put as legend. The default is 'unfolded'.
        show_legend : bool, optional
            If show legend or not. The default is True.
        plotSC : bool, optional
            If to plot the supercell band structure. The default is True.
        legend_pos : TYPE, optional
            Position of the legend. The default is 1.

        Returns
        -------
        ax : matplotlib.pyplot axis
            Figure axis to plot on. 

        """
        if plotSC:
            ax.scatter(data_4_plot[:, 0], data_4_plot[:, 1], s=fatfactor, 
                       color='gray', label="supercell")
        ax.scatter(data_4_plot[:, 0], data_4_plot[:, 1], s=data_4_plot[:, 2]*fatfactor, 
                   marker=marker, color=scatter_color, label=legend_label)
        if show_legend: 
            ax.legend(loc=legend_pos)
        return ax
     
    @classmethod              
    def _plot_density(cls, data_4_plot, ax, Emin, Emax, nE, kpath_in_angs_, 
                      smear, cmap='viridis'):      
        """
        Plot density plot of band structure.

        Parameters
        ----------
        data_4_plot : numpy 2d array
            Data to plot.
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on. 
        Emin : float
            Minimum in energy..
        Emax : float
            Maximum in energy.
        nE : int, 
            Number of pixels in Energy scale when used 'density' mode. 
        kpath_in_angs_ : array
            k on path (in A^-1) coordinate. 
        smear : float
            Gaussian smearing.
        cmap : str/ matplotlib colormap
            Colormap for density plot.The default is 'viridis'.

        Returns
        -------
        ax : matplotlib.pyplot axis
            Figure axis to plot on. 

        """
        energy = np.linspace(Emin, Emax, nE)
        density = np.zeros((len(kpath_in_angs_),nE), dtype=float)
        for k, E, w in data_4_plot[:, :3]:
            ik = np.argmin(abs(k - kpath_in_angs_))
            # Gaussian smearing
            density[ik, :] += w*np.exp(-(energy-E)**2/(2*smear**2)) 
        # density=np.log(density)
        # density[density<1e-3]=0           
        k1 = np.hstack(([kpath_in_angs_[0]], 
                        (kpath_in_angs_[1:]+kpath_in_angs_[:-1])/2,
                        [kpath_in_angs_[-1]]))
        E1 = np.hstack(([energy[0]],(energy[1:]+energy[:-1])/2,[energy[-1]]))
        # density=np.pad(density,((0,1),(0,1)))
        # print("before",k1.shape,E1.shape,density.shape)
        k1, E1 = np.meshgrid(k1,E1)
        # print("after",k1.shape,E1.shape,density.shape)
        plt.pcolormesh(k1, E1, density.T, cmap=cmap)
        plt.colorbar()
        return ax
     
    @classmethod          
    def _plot_band_centers(cls, data_4_plot, ax, color='k', plot_colormap:bool=True,
                           err_bar_fmt='x', color_map='viridis'):
        """
        Plot the band centers and band width.

        Parameters
        ----------
        data_4_plot : numpy 2d array
            Data to plot.
        ax : matplotlib.pyplot axis, optional
            Figure axis to plot on.
        color : matplotlib color/str, optional
            Color of the plots. The default is 'k'.
        plot_colormap : bool, optional
            If to plot the errorbars as colormap. The default is True.
        err_bar_fmt : matplotlib errorbar fmt, optional
            matplotlib errorbar fmt. The default is 'x'.
        color_map : str/ matplotlib colormap
            Colormap for density plot.The default is 'viridis'.

        Returns
        -------
        ax : matplotlib.pyplot axis
            Figure axis to plot on. 

        """
        if plot_colormap:
            # This is super slow
            max_weight, min_weight = data_4_plot[:, -1].max(), data_4_plot[:, -1].min()
            
            cmap = plt.get_cmap(color_map)  
            norm = plt.Normalize(min_weight, max_weight)  # Normalize the float values
            error_colors = cmap(norm(data_4_plot[:, -1]))  # Map float values to colors
                
            for ii, JJ in enumerate(data_4_plot):
                ax.errorbar(JJ[0], JJ[1], yerr=JJ[2], fmt=err_bar_fmt, 
                            ecolor=error_colors[ii], color=error_colors[ii])
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label('Weight')
        else:
            ax.errorbar(data_4_plot[:,0], data_4_plot[:,1], yerr=data_4_plot[:, 2], 
                        fmt=err_bar_fmt, color=color)
        return ax