import numpy as np
import matplotlib.pyplot as plt
from ..BasicFunctions.general_plot_functions import GeneratePlots

### ===========================================================================



class EBSplot(GeneratePlots):
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
        save_figure_dir : TYPE, optional
            Directory where to save the figure. The default is current directory.

        Returns
        -------
        None.

        """
        super().__init__(save_figure_dir=save_figure_dir)
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
        

    def plot(self, save_file_name=None, CountFig=None, Ef=None, Emin=None, Emax=None, 
             pad_energy_scale:float=0.5, mode:str="fatband", yaxis_label:str='E (eV)', 
             special_kpoints:dict=None, plotSC:bool=True, fatfactor=20, nE:int=100, 
             smear:float=0.05, scatter_color='gray', color_map='viridis'):
        """
        Scatter/density plot of the band structure.

        Parameters
        ----------
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
        mode : ['fatband','density'], optional
            Mode of plot. The default is "fatband".
        yaxis_label : str, optional
            Y-axis label text. The default is 'E (eV)'.
        special_kpoints : dictionary, optional
            Dictionary of special kpoints position and labels. If None, ignore
            special kpoints. The default is None.
        plotSC : bool, optional
            Plot supercell bandstructure. The default is True.
        fatfactor : int, optional
            Scatter plot marker size. The default is 20.
        nE : int, optional
            Number of pixels in Energy scale when used 'density' mode. 
            The default is 100.
        smear : float, optional
            Gaussian smearing. The default is 0.05.
        scatter_color : str/color, optional
            Color of scatter plot of unfolded band structure. The color of supercell
            band structures is gray. The default is 'gray'.
        color_map: str/ matplotlib colormap
            Colormap for density plot. The default is viridis.
        
        Raises
        ------
        ValueError
            If plot mode is unknown.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure instance.
        ax : Axis instance
            Figure axis instance.
        CountFig: int or None
            Figure count.

        """
        
        print('- Plotting band structures...')
        if Ef == 'auto' or Ef is None: 
            Ef = self.efermi
        # Shift the energy scale to 0 fermi energy level   
        if Ef is not None:
            self.plot_result[:,1] -= Ef 
            yaxis_label = r"E$-$E$_\mathrm{F}$ (eV)"
            print(f"-- Efermi was set to {Ef} eV")
            
        fig, ax = plt.subplots()
        ax.set_ylabel(yaxis_label)
        
        if Emin is None: Emin = self.plot_result[:,1].min() - pad_energy_scale
        if Emax is None: Emax = self.plot_result[:,1].max() + pad_energy_scale
        
        result = self.plot_result[(self.plot_result[:,1] >= Emin - max(smear*10, 0.1)) * 
                                  (self.plot_result[:,1] <= Emax + max(smear*10, 0.1))]
        
        # Plot as fat band
        if mode=="fatband":
            if plotSC:
                ax.scatter(result[:, 0], result[:, 1], s=fatfactor, color='gray', label="supercell")
            ax.scatter(result[:, 0], result[:, 1], s=result[:, 2]*fatfactor, color=scatter_color, label="unfolded")
            ax.legend(loc=1)
        elif mode=="density":
            energy = np.linspace(Emin, Emax, nE)
            density = np.zeros((len(self.kpath_in_angs_),nE), dtype=float)
            for k, E, w in result[:, :3]:
                ik = np.argmin(abs(k - self.kpath_in_angs_))
                # Gaussian smearing
                density[ik, :] += w*np.exp(-(energy-E)**2/(2*smear**2)) 
            # density=np.log(density)
            # density[density<1e-3]=0           
            k1 = np.hstack(([self.kpath_in_angs_[0]], 
                            (self.kpath_in_angs_[1:]+self.kpath_in_angs_[:-1])/2,
                            [self.kpath_in_angs_[-1]]))
            E1 = np.hstack(([energy[0]],(energy[1:]+energy[:-1])/2,[energy[-1]]))
            # print(k1,E1)
            # density=np.pad(density,((0,1),(0,1)))
            # print("before",k1.shape,E1.shape,density.shape)
            k1, E1 = np.meshgrid(k1,E1)
            # print("after",k1.shape,E1.shape,density.shape)
            plt.pcolormesh(k1, E1, density.T, cmap=color_map)
            plt.colorbar()
        else:
            raise ValueError("Unknownplot mode: '{}'".format(mode))
            
        if special_kpoints is not None:
            kl = np.array([self.kpath_in_angs_[ik] for ik in special_kpoints.keys()])
            ll = np.array([k for k in special_kpoints.values()])
            borders = [0] + list(np.where((kl[1:]-kl[:-1])>1e-4)[0]+1) + [len(kl)]
            k_labels=[(kl[b1:b2].mean(),"/".join(set(ll[b1:b2]))) for b1,b2 in zip(borders,borders[1:])]
            
            x_tiks_labels = [label[1] for label in k_labels]
            x_tiks_positions = [label[0] for label in k_labels]
            plt.xticks(x_tiks_positions, x_tiks_labels)
            # Draw vertical lines
            for label in k_labels:
                plt.axvline(x=label[0], color='k', lw=2)
                
        ax.set_ylim([Emin, Emax])
        ax.set_xlim([self.kpath_in_angs_.min(), self.kpath_in_angs_.max()])

        if save_file_name is None:
            plt.show()
        else:
            CountFig = self.save_figure(save_file_name, CountFig=CountFig)
            plt.close()
        return fig, ax, CountFig

