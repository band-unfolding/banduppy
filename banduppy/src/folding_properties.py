import numpy as np
from ..BasicFunctions.general_functions import _draw_line_length
    
### ===========================================================================
class FindProperties:
    def __init__(self):
        pass
    
    @property
    def nkptSBZ(self):
        '''
        Return number of (unique) SC kpoints.
        '''
        return(self.SBZ_kpts_list.shape[0])
    
    @property
    def nkptPBZ(self):
        '''
        Return number of PC kpoints.
        '''
        return(self.PBZ_kpts_list_org.shape[0])

    @property
    def nkptPBZ_unique(self):
        '''
        Return number of PC kpoints.
        '''
        return(self.PBZ_kpts_list.shape[0])
    
    def _print_info(self, level='low'):
        """
        Printing information about the band folding.
        
        Percentage folding = ((#of PC kpoints - #of folded SC Kpoints)/(#of PC kpoints))*100

        Parameters
        ----------
        level : ['low','medium','high'], optional
            Level of printing information. The default is 'low'.

        Returns
        -------
        None.

        """
        print(f"{'='*_draw_line_length}\n- Folding info:")
        print(f"-- Total number of unique PC kpoints: {self.nkptPBZ} ({self.nkptPBZ_unique} unique)")
        print(f"-- {self.nkptPBZ_unique} unique PC k-points => {self.nkptSBZ} unique SC K-points")
        print(f'-- Percentage folding: {(self.nkptPBZ_unique-self.nkptSBZ)/self.nkptPBZ_unique*100:.3f} %')
        if level in ['medium','high']:
            print(f"{'='*_draw_line_length}")
            print(f'- Special k-points (pos_index, label): {self.special_kpoints_pos_labels}')
            if level == 'high':
                self._print_PBZ_SBZ_kpts_mapping_full()
            else:
                self._print_PBZ_SBZ_kpts_mapping()

    def kpoints_SBZ_str(self):
        """
        Create string of SC kpoints generated.

        Returns
        -------
        str
            String of SC kpoints generated..

        """
        return f"{self.nkptSBZ}\n"+"\n".join("  ".join(f"{x:12.8f}" for x in k ) 
                                                     for k in self.SBZ_kpts_list)+"\n"
    
    def _get_PBZ_SBZ_kpts_mapping(self):
        """
        Find mapping of SC kpoints to PC kpoints.

        Returns
        -------
        None.

        """
        self.SBZ_PBZ_kpts_mapping = {}
        
        for i in np.unique(self.kpointsSBZ_index_in_unique):
            self.SBZ_PBZ_kpts_mapping[i] = {}
            for mm in np.argwhere(self.kpointsSBZ_index_in_unique==i).flatten():
                self.SBZ_PBZ_kpts_mapping[i][mm] = []
                for kk in np.argwhere(self.kpointsPBZ_index_in_unique==mm).flatten():
                    self.SBZ_PBZ_kpts_mapping[i][mm].append(kk)
        return 
    
    def _print_PBZ_SBZ_kpts_mapping(self):
        """
        Print mapping of SC kpoints to PC kpoints.

        Returns
        -------
        None.

        """
        print(f"{'='*_draw_line_length}\n- K-k relation: (K index: K -> k index: k)")
        for key, val in self.SBZ_PBZ_kpts_mapping.items(): # key: SC-kp index; val: list of unique PC-kpts indices
            print(f"-- {key:>5}:" + "  ".join(f"{x:12.8f}" for x in self.SBZ_kpts_list[key][:3]))
            for _, vall in val.items(): # kkk: unique PC-kpts indices; vall: list of PC-kpts indices
                    for kk in vall:
                            print(f"\t--- {kk:>5}:" + "  ".join(f"{x:12.8f}" for x in self.PBZ_kpts_list_org[kk, :3]))
    
    def _print_PBZ_SBZ_kpts_mapping_full(self):
        """
        Print mapping of SC kpoints to PC kpoints.

        Returns
        -------
        None.

        """
        print(f"{'='*_draw_line_length}\n- K-k relation: (K index: K -> k index unique: k unique -> k index: k)")
        for key, val in self.SBZ_PBZ_kpts_mapping.items(): # key: SC-kp index; val: list of unique PC-kpts indices
            print(f"-- {key:>5}:" + "  ".join(f"{x:12.8f}" for x in self.SBZ_kpts_list[key, :3]))
            for kkk, vall in val.items(): # kkk: unique PC-kpts indices; vall: list of PC-kpts indices
                    print(f"\t--- {kkk:>5}:" + "  ".join(f"{x:12.8f}" for x in self.PBZ_kpts_list[kkk, :3]))
                    for kk in vall:
                            print(f"\t\t---- {kk:>5}:" + "  ".join(f"{x:12.8f}" for x in self.PBZ_kpts_list_org[kk, :3]))
