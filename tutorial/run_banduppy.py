#!/usr/bin/env python3

### you need to install packages `banduppy` and `irrep` from pip

import banduppy

import shutil,os,glob
from subprocess import run
import numpy as np
import pickle

QEpath="/app/theorie/qe-6.4.1/bin"

nproc=16
PWSCF="mpirun -np {np}  {QEpath}/pw.x -nk {npk} -nb {npb}".format(np=nproc,QEpath=QEpath,npk=nproc/4,npb=4).split()


unfold=banduppy.UnfoldingPath(
            supercell= [[-1 ,  1 , 1],
                        [1  , -1 , 1],
                        [1  ,  1 ,-1]] ,   # How the SC latticevectors are expressed in the PC basis (should be a 3x3 array of integers)
            pathPBZ=[[1/2,1/2,1/2], [0,0,0],[1/2,0,1/2], [5/8,1/4,5/8], None, [3/8,3/8,3/4], [0,0,0]],  # Path nodes in reduced coordinates in the primitive BZ. if the segmant is skipped, put a None between nodes
            nk=(23,27,9,29),  #  number of k-points in each non-skipped segment. Or just give one number, if they are equal
             labels="LGXUKG" )   # or ['L','G','X','U','K','G']

kpointsPBZ=unfold.kpoints_SBZ_str()   # as tring  containing the k-points to beentered into the PWSCF input file  after 'K_POINTS crystal ' line. maybe transformed to formats of other codes  if needed

try:
    print ("unpickling unfold")
    unfold=pickle.load(open("unfold.pickle","rb"))
except Exception as err:
    print("error while unpickling unfold '{}',  unfolding it".format(err))
    try:
        print ("unpickling bandstructure")
        bands=pickle.load(open("bandstructure.pickle","rb"))  
        print ("unpickling - success")
    except Exception as err:
        print("Unable to unpickle  bandstructure '{}' \n  Reading bandstructurefrom .save folder ".format(err))
        try: 
            ####   This line reads the bandstructure written by QE into an pobject bandStructure of the irrep code.
            bands=banduppy.BandStructure(code="espresso", prefix="bulk_Si")
            ####  this is a shortcut to   bands=irrep.bandstructure.BandStructure(...)
            ####  For spin-polarised calculations you need to select spin channel 'up' or 'dw' 
            ####   (works only with QE, and you need irrep>=1.5.3 installed. Example:
            ####     bands=banduppy.BandStructure(code="espresso", prefix="bulk_Si",spin_channel='up')   # or 'dw'
            ####   examples for other codes are:
            ####  VASP:
            ####      bands=banduppy.BandStructure(fWAV='WAVECAR',fPOS='POSCAR',spinor=False,code='vasp')
            ####  Abinit:
            ####      bands=banduppy.BandStructure(fWFK='mysystem_WFK',code='abinit')
            ####  Files preparedfor Wannier90 (.eig, .win, UNK* ):
            ####      bands=banduppy.BandStructure(prefix='nbcosb',code='wannier90')
            #####   Other parameters may be used (..,Ecut=.,IBstart=..,IBend=..,kplist=..,EF=..)

        except Exception as err:
            print("error reading  bandstructure '{}' \n calculating it".format(err))
            pw_file="bulk_Si"
            shutil.copy("input/"+pw_file+"-scf.in",".")
            open(pw_file+"-bands.in","w").write(open("input/"+pw_file+"-bands.in").read()+kpointsPBZ)
            scf_run=run(PWSCF,stdin=open(pw_file+"-scf.in"),stdout=open(pw_file+"-scf.out","w"))
            bands_run=run(PWSCF,stdin=open(pw_file+"-bands.in"),stdout=open(pw_file+"-bands.out","w"))
            for f in glob.glob("*.wfc*"):
                os.remove(f)
        bands=banduppy.BandStructure(code="espresso", prefix="bulk_Si")
        pickle.dump(bands,open("bandstructure.pickle","wb"))

    unfold.unfold(bands,break_thresh=0.1)
    pickle.dump(unfold,open("unfold.pickle","wb"))

#now plot the result as fat band
unfold.plot(save_file="unfold_fatband.png",plotSC=True,Emin=-5,Emax=5,Ef='auto',fatfactor=50,mode='fatband') 
#or as a colormap
unfold.plot(save_file="unfold_density.png",plotSC=True,Emin=-5,Emax=5,Ef='auto',mode='density',smear=0.2,nE=200) 

#or use the data to plot in any other format
data=np.loadtxt("bandstructure_unfolded.txt")



