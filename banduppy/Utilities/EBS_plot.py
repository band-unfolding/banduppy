import numpy as np
import matplotlib.pyplot as plt
from ..BasicFunctions.general_plot_functions import _GeneratePlots
from .EBS_properties import _GeneralFunctionsDefs, _FormatSpecialKpts

### ===========================================================================
class _EBSplot(_GeneratePlots, _GeneralFunctionsDefs, _FormatSpecialKpts):
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
            Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy, weight, "Sx, Sy, Sz" if spinor.] or
            Format: [kpoint coordinate, Band center, Band width, Sum of dN] for band centers
            The default is None.
        save_figure_dir : str/path, optional
            Directory where to save the figure. The default is current directory.
            
        Returns
        -------
        kpath_in_angs : array
            k on path (in A^-1) coordinate.
        unfolded_bandstructure : ndarray
            Unfolded effective band structure/band center data. 
            Format: [k index, k on path (A^-1), energy, weight] or
            Format: [kpoint coordinate, Band center, Band width, Sum of dN] for band centers
        efermi : float
            Default Fermi energy. Set to 0.0.

        """
        _GeneratePlots.__init__(self, save_figure_dir=save_figure_dir)
        self.efermi = 0.0
        
        if kpath_in_angs is None:
            try:
                self.kpath_in_angs_ = self.kpline.copy()
            except:
                raise ValueError('Provide k-path data (kpath_in_angs).')
        else:
            self.kpath_in_angs_ = kpath_in_angs.copy()
        
        if unfolded_bandstructure is None:
            try:
                plt_result = self.unfolded_bandstructure.copy() 
            except:
                raise ValueError('No bandstructure data file is found')
        else:
            plt_result = unfolded_bandstructure.copy() 
            
        self.plot_result = _GeneralFunctionsDefs._reformat_columns_full_bandstr_data(plt_result)
    
    def _plot(self, fig=None, ax=None, save_file_name=None, CountFig=None, Ef=None, Emin=None, 
              Emax=None,  pad_energy_scale:float=0.5, threshold_weight:float=None,  
              mode:str="fatband", yaxis_label:str='E (eV)', special_kpoints:dict=None, 
              plotSC:bool=True, marker='o', fatfactor=20, nE:int=100, smear:float=0.05,  
              color='gray', color_map='viridis', plot_colormap_bandcenter:bool=True,
              show_legend:bool=True, show_colorbar:bool=False, colorbar_label:str=None,
              vmin=None, vmax=None, show_plot:bool=True, **kwargs_savefig):
        """
        Scatter/density/band_centers plot of the band structure.

        Parameters
        ----------
        fig : matplotlib.pyplot figure instance, optional
            Figure instance to plot on. The default is None.
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
        show_colorbar : bool, optional
            Plot the colorbar in the figure or not. If fig=None, this is ignored.
            The default is False.
        colorbar_label : str, optional
            Colorbar label. The default is None. If None, ignored.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers. 
            By default, the colormap covers the complete value range of the supplied data.
        show_plot : bool, optional
            To show the plot when not saved. The default is True.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.
        
        Raises
        ------
        ValueError
            If plot mode is unknown.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure instance. If ax is not None previously generated/passed fig instance
            will be returned. Return None, if no fig instance is inputed along with ax.
        ax : Axis instance
            Figure axis instance.
        CountFig: int or None
            Figure count.

        """
        if ax is None: 
            self.fig, ax = plt.subplots()
        else:
            self.fig = fig 
            
        if yaxis_label is None: yaxis_label=''
        
        if mode != "band_centers" and len(self.plot_result[0]) > 3: 
            self.plot_result = self.plot_result[:, 1:]
        
        if Ef == 'auto' or Ef is None:  Ef = self.efermi
            
        Emin, Emax, result = \
            _GeneralFunctionsDefs._get_data_in_energy_window(self.plot_result, 
                                                             Ef, Emin=Emin, Emax=Emax,  
                                                             pad_energy_scale=pad_energy_scale, 
                                                             threshold_weight=threshold_weight)
        # Shift the energy scale to 0 fermi energy level   
        if Ef is not None:
            ax.axhline(y=0, color='k', ls='--', lw=1)
            if yaxis_label == 'E (eV)': yaxis_label = r"E$-$E$_\mathrm{F}$ (eV)"
        
        # Plot as fat band
        if mode == "fatband":
            ax, return_plot = self._plot_fatband(result, ax, marker=marker, fatfactor=fatfactor, 
                                                 scatter_color=color, show_legend=show_legend,
                                                 plotSC=plotSC)
        elif mode == "density":
            
            ax, return_plot = self._plot_density(result, ax, Emin, Emax, nE, self.kpath_in_angs_, 
                                                 smear, cmap=color_map, vmin=vmin, vmax=vmax)
        elif mode == 'band_centers':
            ax, return_plot = self._plot_band_centers(result, ax, color=color, color_map=color_map,
                                                      plot_colormap=plot_colormap_bandcenter,
                                                      min_weight=vmin, max_weight=vmax)
        elif mode == 'only_for_all_scf': # This mode is hidden. Used for all_scf plots later.
            pass # This plots the skeleton of the plots without raising error.
        else:
            raise ValueError("Unknownplot mode: '{}'".format(mode))
            
        if show_colorbar and (self.fig is not None):
            cbar = self.fig.colorbar(return_plot, ax=ax)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

        if special_kpoints is not None:
            x_tiks_labels, x_tiks_positions = \
                _FormatSpecialKpts._extract_special_kpts_info(special_kpoints, 
                                                              self.kpath_in_angs_)
            ax.set_xticks(x_tiks_positions, x_tiks_labels)
            # Draw vertical lines
            for line_pos in x_tiks_positions:
                ax.axvline(x=line_pos, color='k', ls='--', lw=2)
        
        ax.set_ylabel(yaxis_label)
        ax.set_ylim([Emin, Emax])
        ax.set_xlim([self.kpath_in_angs_.min(), self.kpath_in_angs_.max()])

        if save_file_name is None:
            if show_plot: plt.show()
        else:
            CountFig = self._save_figure(save_file_name, fig=self.fig, CountFig=CountFig, **kwargs_savefig)
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
        sc : figure instance

        """
        if plotSC:
            sc = ax.scatter(data_4_plot[:, 0], data_4_plot[:, 1], s=fatfactor, 
                            color='gray', label="supercell")
        sc = ax.scatter(data_4_plot[:, 0], data_4_plot[:, 1], s=data_4_plot[:, 2]*fatfactor, 
                        marker=marker, color=scatter_color, label=legend_label)
        if show_legend: 
            ax.legend(loc=legend_pos)
        return ax, sc
     
    @classmethod              
    def _plot_density(cls, data_4_plot, ax, Emin, Emax, nE, kpath_in_angs_, 
                      smear, cmap='viridis', vmin=None, vmax=None):      
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
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers. 
            By default, the colormap covers the complete value range of the supplied data.

        Returns
        -------
        ax : matplotlib.pyplot axis
            Figure axis to plot on. 
        pcm : matplotlib pcolormesh plot instance
            Plot instance.

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
        pcm = ax.pcolormesh(k1, E1, density.T, cmap=cmap, vmin=vmin, vmax=vmax)
        return ax, pcm
     
    @classmethod          
    def _plot_band_centers(cls, data_4_plot, ax, color='k', plot_colormap:bool=True,
                           err_bar_fmt='x', color_map='viridis', min_weight:float=None,
                           max_weight:float=None):
        """
        Plot the band centers and band width.

        Parameters
        ----------
        data_4_plot : numpy 2d array
            Data to plot.
            Format: [kpoint coordinate, Band center, Band width, Sum of dN]
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
        min_weight : float, optional
            Minimum in the color scale for the band centers. 
            The default is None. If None, determined from the data array supplied.
        max_weight : float, optional
            Maximum in the color scale for the band centers.
            The default is None. If None, determined from the data array supplied.

        Returns
        -------
        ax : matplotlib.pyplot axis
            Figure axis to plot on. 
        cmap_mappable :
            Figure instance or colormap instance for colorbar.

        """
        if plot_colormap:
            # This is super slow
            if min_weight is None:
                min_weight = data_4_plot[:, -1].min()
            if max_weight is None:
                max_weight = data_4_plot[:, -1].max()
            
            cmap = plt.get_cmap(color_map)  
            norm = plt.Normalize(min_weight, max_weight)  # Normalize the float values
            error_colors = cmap(norm(data_4_plot[:, -1]))  # Map float values to colors
                
            for ii, JJ in enumerate(data_4_plot):
                ax.errorbar(JJ[0], JJ[1], yerr=JJ[2], fmt=err_bar_fmt, 
                            ecolor=error_colors[ii], color=error_colors[ii])
            cmap_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            cmap_mappable = ax.errorbar(data_4_plot[:,0], data_4_plot[:,1], yerr=data_4_plot[:, 2], 
                                        fmt=err_bar_fmt, color=color, linestyle='')
        return ax, cmap_mappable
    
    def _plot_scf(self, al_scf_data, plot_max_scf_steps:int=None, 
                  save_file_name=None, Ef=None, Emin=None, 
                  Emax=None,  pad_energy_scale:float=0.5, threshold_weight:float=None,  
                  yaxis_label:str='E (eV)', special_kpoints:dict=None, 
                  plot_sc_unfold:bool=True, marker='o', fatfactor=20, smear:float=0.05,  
                  color='gray', color_map='viridis', plot_colormap_bandcenter:bool=True,
                  show_legend:bool=True, show_colorbar:bool=True, colorbar_label:str=None,
                  vmin=None, vmax=None, show_plot:bool=True, **kwargs_savefig):
        """
        Band centers all scf steps plot.

        Parameters
        ----------
        al_scf_data : dictionary
            All SCF data.
            Each array contains the final details of band centers in a particular
            kpoint. The dictionary then contains the details for each SCF cycles with
            keys are the SCF cycle number. The highest level dictionary then contains 
            details for each kpoints with keys are the kpoint indices. Returns None
            if collect_data_scf is false.
            Format: {kpoint_index: {SCF_cycle_index: [Band center, Band width, Sum of dN]}}
        plot_max_scf_steps : int, optional
            How many maximum scf cycle to plot?
            The default is maximum SCF steps found in the dictionary of all k-points.
            If scf cycle not found for a particular kpoint previous SCF cycle will be plotted.
        save_file_name : str, optional
            Name of the figure file. If None, figure will be not saved. 
            The default is None.
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
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plot_sc_unfold : bool, optional
            Plot supercell unfolded bandstructure. The default is True.
        marker : matplotlib.pyplot markerMarkerStyle, optional
            The marker style. Marker can be either an instance of the class or 
            the text shorthand for a particular marker. 
            The default is 'o'.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        color : str/color, optional
            Color for band centers plot when color_map is not used. 
            The default is 'gray'. The color of supercell
            band structures is gray always. 
        color_map: str/ matplotlib colormap
            Colormap for band centers plot. The default is viridis.
        plot_colormap_bandcenter : bool, optional
            If plotting the band ceneters by colormap. The default is True.
        show_legend : bool, optional
            If show legend or not. The default is True.
        show_colorbar : bool, optional
            Plot the colorbar in the figure or not. If fig=None, this is ignored.
            The default is True.
        colorbar_label : str, optional
            Colorbar label. The default is None. If None, ignored.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers. 
            By default, the colormap covers the complete value range of the supplied data.
        show_plot : bool, optional
            To show the plot when not saved. The default is True.
        **kwargs_savefig : dict
            The matplotlib keywords for savefig function.
        
        Raises
        ------
        ValueError
            If plot mode is unknown.

        """
        # If scf cycle not found for a particular kpoint previous SCF cycle will be plotted.
        if plot_max_scf_steps is None:
            plot_max_scf_steps = max([max(al_scf_data[JJ].keys()) for JJ in al_scf_data.keys()])
        elif plot_max_scf_steps <1 : 
            print('Warning: scf steps indices starts with 1.')
            plot_max_scf_steps = 1
        else:
            max_scf_steps_f = max([max(al_scf_data[JJ].keys()) for JJ in al_scf_data.keys()])
            if plot_max_scf_steps > max_scf_steps_f:
                _rtx__ = f'Requested {plot_max_scf_steps} maximum SCF steps plot. Found {max_scf_steps_f} maximum SCF steps in dictionary.'
                print(f'Warning: {_rtx__}. Will plot {max_scf_steps_f} SCF steps.')
                plot_max_scf_steps = max_scf_steps_f
        
        # Find the maximum and minimum in color scale
        if plot_colormap_bandcenter:
            minN_weight, maxX_weight = 0, 0
            for which_kp in al_scf_data.values():
                for ZZ_tmp in which_kp.values():
                    ZZ= ZZ_tmp[:, -1]
                    max_weight, min_weight = ZZ.max(), ZZ.min()
                    if max_weight > maxX_weight:
                        maxX_weight = max_weight
                    if min_weight < minN_weight:
                        minN_weight = min_weight
            if vmin is None:
                vmin = minN_weight
            if vmax is None:
                vmax = maxX_weight
            
        plot_mode = 'fatband' if plot_sc_unfold else 'only_for_all_scf'
        if Ef == 'auto' or Ef is None: Ef = self.efermi
        count_fig_pec = len(str(plot_max_scf_steps+1))
        
        for scf_step in range(1, plot_max_scf_steps+1): # loop over scf cycle
            print(f'-- Plotting SCF step: {scf_step}')
            fig, ax = plt.subplots()
            fig, ax, _ \
            = self._plot(fig=fig, ax=ax, save_file_name=None, Ef=Ef, 
                        Emin=Emin, Emax=Emax, pad_energy_scale=pad_energy_scale, 
                        threshold_weight=threshold_weight, mode=plot_mode,
                        yaxis_label=yaxis_label, special_kpoints=special_kpoints, 
                        plotSC=False, marker=marker, fatfactor=fatfactor, 
                        smear=smear, color='gray', show_legend=show_legend, 
                        show_colorbar=False, show_plot=False)
            save_file_name_ = f'{scf_step:0{count_fig_pec}}_{save_file_name}'
            for which_kp in al_scf_data: # loop over k-points
                while True:
                    if al_scf_data[which_kp].get(scf_step) is None:
                        scf_step -= 1
                        #print(f'-- Going back to previous SCF step for this kpoint: {scf_step}')
                    else:
                        break

                XX = [self.kpath_in_angs_[which_kp]]*len(al_scf_data[which_kp][scf_step])
                YY = al_scf_data[which_kp][scf_step][:, 0] - Ef
                result_tmp = np.column_stack( (XX, YY, al_scf_data[which_kp][scf_step][:, 1:]) )
                
                ax, return_plot = self._plot_band_centers(result_tmp, ax, color=color, 
                                                          color_map=color_map,
                                                          plot_colormap=plot_colormap_bandcenter,
                                                          min_weight=vmin,
                                                          max_weight=vmax)
                
            if plot_colormap_bandcenter and show_colorbar:
                cbar = fig.colorbar(return_plot, ax=ax)
                if colorbar_label is not None:
                    cbar.set_label(colorbar_label)

            # ax.set_ylim([Emin, Emax])
            # ax.set_xlim([self.kpath_in_angs_.min(), self.kpath_in_angs_.max()])

            if save_file_name is None:
                if show_plot: plt.show()
            else:
                _ = self._save_figure(save_file_name_, fig=fig, CountFig=None, **kwargs_savefig)
                plt.close()