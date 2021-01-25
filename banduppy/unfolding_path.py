import numpy as np
import irrep
from irrep.__aux import is_round
from  irrep.bandstructure import BandStructure
assert irrep.__version__ >="1.5"
from collections import Iterable

class UnfoldingPath():

    def __init__(self,supercell=np.eye(3,dtype=int),pathPBZ=[],nk=11,labels=None):
        if isinstance(nk, Iterable):
            nkgen=(x for x in nk)
        else:
            nkgen=(nk for x in pathBZ)
        supercell_int=np.round(supercell)
        assert supercell_int.shape==(3,3), "supercell should be 3x3, founf {}".format(supercell_int.shape)
        assert np.linalg.norm(np.array(supercell)-supercell_int)<1e-14 , "supercell should consist of integers, found {}".format(supercell)
        self.supercell=np.array(supercell_int,dtype=int)
        assert np.linalg.det(self.supercell)!=0, "the supercell vectors should be linear independent"
        self.kpointsPBZ=np.zeros((0,3))
        self.i_labels={}
        self.nodes=[k for k in pathPBZ if k is not None]
        if labels is None:
            labels=[str(i+1) for i,k in enumerate(self.nodes)]
        labels=(l for l in labels)
        labels=[None if k is None else next(labels)  for k in pathPBZ]
        
#        print (pathPBZ,pathPBZ[1:],labels,labels[1:])
        for start,end,l1,l2 in zip(pathPBZ,pathPBZ[1:],labels,labels[1:]) :
            if None not in (start,end):
                self.i_labels[self.kpointsPBZ.shape[0]]=l1
                start=np.array(start)
                end=np.array(end)
                assert start.shape==end.shape==(3,)
                self.kpointsPBZ=np.vstack( (self.kpointsPBZ,start[None,:]+np.linspace(0,1.,next(nkgen))[:,None]*(end-start)[None,:] ) )
                self.i_labels[self.kpointsPBZ.shape[0]-1]=l2
        self.kpointsPBZ_unique=np.unique(self.kpointsPBZ%1,axis=0)
        self.kpointsPBZ_index_in_unique=np.array([np.where( (self.kpointsPBZ_unique==kp%1).prod(axis=1)   )[0][0] for kp in self.kpointsPBZ])
        kpointsSBZ=self.kpointsPBZ_unique.dot(self.supercell.T)%1
        kpointsSBZ_unique=np.unique(kpointsSBZ%1,axis=0)
        self.kpointsPBZ_unique_index_SBZ=np.array([np.where( (kpointsSBZ_unique==kp).prod(axis=1)   )[0][0] for kp in kpointsSBZ])
        self.kpointsSBZ=kpointsSBZ_unique

    @property
    def nkptSBZ(self):
        return(self.kpointsSBZ.shape[0])

    @property
    def nkptPBZ(self):
        return(self.kpointsPBZ.shape[0])

    @property
    def nkptPBZunique(self):
        return(self.kpointsPBZ_unique.shape[0])

    def kpoints_SBZ_str(self):
        return "{}\n".format(self.nkptSBZ)+"\n".join("  ".join("{0:12.8f}".format(x) for x in k )+"    1" for k in self.kpointsSBZ   )+"\n"

    def print_PBZ_SBZ(self):
        print ("contains {} points in PC ({} unique),corresponding to {} unique SC points".format(self.nkptPBZ, self.kpointsPBZ_unique.shape[0], self.nkptSBZ) )
        for i,kp in enumerate(self.kpointsPBZ):
           j= self.kpointsPBZ_unique_index_SBZ[self.kpointsPBZ_index_in_unique[i]]
           print (i,kp,j,self.kpointsSBZ[j])
           
    @property
    def path_str(self):
        result=["nodes of the path: "]
        for kl,n in zip(self.k_labels,self.nodes):
#            print (kl,n)
            result.append("{:10.6f} {:8s} {:12.8f} {:12.8f} {:12.8f}".format(kl[0],kl[1],n[0],n[1],n[2]))
        return "".join("# "+l+"\n" for l in result)

    def unfold(self,bandstructure,break_thresh=0.1):
     #  first evaluate the path as line ad store it
        self.kpline=bandstructure.KPOINTSline(kpred=self.kpointsPBZ,breakTHRESH=break_thresh)
        self.efermi=bandstructure.efermi
        k_labels=[(self.kpline[ik],lab) for ik,lab in self.i_labels.items()]
        ll=np.array([k[1] for k in k_labels])
        kl=np.array([k[0] for k in k_labels])
        borders=[0]+list(np.where((kl[1:]-kl[:-1])>1e-4)[0]+1)+[len(kl)]
#        print ("kl=",kl,"\nborders=",borders)
        self.k_labels=[(kl[b1:b2].mean(),"/".join(set(ll[b1:b2]))) for b1,b2 in zip(borders,borders[1:])]




        kpSBZcalc={}
        for ik,kpSBZ in enumerate(self.kpointsSBZ):
            found=False
            for KP in bandstructure.kpoints:
                if is_round(KP.K-kpSBZ,prec=1e-6):
                    kpSBZcalc[ik]=KP
                    found=True
            if not found:
                print ("WARNING: k-point {} was not found in the calculated bandstructure. the corresponding points in the unfolding path will be skipped".format(kpSBZ))
        unfolded_unique={}
        for ik,kpPBZu in enumerate(self.kpointsPBZ_unique):
            jk=self.kpointsPBZ_unique_index_SBZ[ik]
            if jk in kpSBZcalc:
                unfolded_unique[ik]=kpSBZcalc[jk].unfold(supercell=self.supercell,kptPBZ=kpPBZu)
        unfolded_found={}
        for ik,kpt in enumerate(self.kpointsPBZ):
            jk=self.kpointsPBZ_index_in_unique[ik]
            if jk in unfolded_unique:
                unfolded_found[ik]=unfolded_unique[jk]
            else:
                print ("WARNING: no SBZ point found to unfold on the PBZ k point {}. Skipping... ".format(kpt) )
        indices_found=list(unfolded_found.keys())

        with open("kpath_unfolded.txt","w") as fpath:
            fpath.write(self.path_str)
            np.savetxt(fpath, np.hstack( (self.kpline[:,None],self.kpointsPBZ))[indices_found],header="# k on path (A^-1) and reduced coordinates k1,k2,k3")


        result=[]
        for ik,kpl in enumerate(self.kpline):
            if ik in unfolded_found:
                for band in unfolded_found[ik] :
                    result.append([kpl,]+list(band))
        self.result=np.array(result)


        np.savetxt("bandstructure_unfolded.txt", result ,header="# k on path (A^-1) ,energy, weight "+("Sx,Sy,Sz" if bandstructure.spinor else "")  +"\n")
        return result



    def plot(self,save_file=None,Ef=None,Emin=None,Emax=None,mode="fatband",plotSC=True,fatfactor=20,nE=100,smear=0.05):
        import matplotlib.pyplot as plt
        result=self.result.copy()
        if Ef=='auto' : 
            Ef=self.efermi
        if Ef is not None:
            result[:,1]-=Ef
            plt.ylabel(r"$E-E_F$, eV")
            print ("Efermi was set to {} eV".format(Ef))
        else:
            plt.ylabel(r"$E$, eV")
        if Emin is None:
            Emin=result[:,1].min()-0.5
        if Emax is None:
            Emax=result[:,1].max()+0.5
        result=result[(result[:,1]>=Emin-max(smear*10,0.1))*(result[:,1]<=Emax+max(smear*10,0.1))]

        if mode=="fatband":
            if plotSC:
                plt.scatter(result[:,0],result[:,1],s=fatfactor,color="gray",label="supercell")
            plt.scatter(result[:,0],result[:,1],s=result[:,2]*fatfactor,color="red",label="unfolded")
            plt.legend()
        elif mode=="density":
            energy=np.linspace(Emin,Emax,nE)
            density=np.zeros((self.nkptPBZ,nE),dtype=float)
            for k,E,w in result[:,:3]:
                ik=np.argmin(abs(k-self.kpline))
#                print (ik,k,E,w)
                density[ik,:]+=w*np.exp( -(energy-E)**2/(2*smear**2))
#            density=np.log(density)
#            density[density<1e-3]=0
            k1=self.kpline.copy()
            k1=np.hstack(([k1[0]],(k1[1:]+k1[:-1])/2,[k1[-1]]))
            E1=np.hstack(([energy[0]],(energy[1:]+energy[:-1])/2,[energy[-1]]))
#            print(k1,E1)
#            density=np.pad(density,((0,1),(0,1)))
#            print("before",k1.shape,E1.shape,density.shape)
            k1,E1=np.meshgrid(k1,E1)
#            print("after",k1.shape,E1.shape,density.shape)
            plt.pcolormesh(k1,E1,density.T)
            plt.colorbar()
        else:
            raise ValueError("Unknownplot mode: '{}'".format(mode))

        x_tiks_labels = [label[1] for label in self.k_labels]
        x_tiks_positions = [label[0] for label in self.k_labels]
        plt.xticks(x_tiks_positions, x_tiks_labels )

        for label in self.k_labels:
            plt.axvline(x=label[0] )
        plt.ylim([Emin,Emax])
        plt.xlim([result[:,0].min(),result[:,0].max()])

        if save_file is None:
           plt.show()
        else:
           plt.savefig(save_file)
        plt.close()

