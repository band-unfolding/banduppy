#!/bin/bash
#PBS -N TestUnfolding
#PBS -l nodes=1:ppn=22
#PBS -l walltime=2:00:00
#PBS -q default
#PBS -o /sfihome/badal.mondal/err/"${PBS_JOBNAME}_${PBS_JOBID}.out"
#PBS -e /sfihome/badal.mondal/err/"${PBS_JOBNAME}_${PBS_JOBID}.err"
##PBS -W x=NACCESSPOLICY:SINGLEJOB

## ====================== Update module path ============================
source /usr/share/Modules/init/bash
module use --append /local/rocks7-temp/modules
module use --append /sfihome/badal.mondal/local/Modulefiles
module purge

## =================== Total number of CPU allocated =====================
echo "Total CPU count = $PBS_NP"

## ========================== Load modules ===============================
module load dftb/condaDFTB
module load vasp/5.4.4

## =================== Set job submit directory ==========================
JOB_DIRECTORY='/sfiwork/badal.mondal/TestUnfolding/tutorials/VASP'
for II in {Si,Ge,SiGe};do
	JJ=${JOB_DIRECTORY}/${II}
	cd ${JJ}/reference_without_SOC
	python ${JJ}/run_banduppy_hpc.py
done

