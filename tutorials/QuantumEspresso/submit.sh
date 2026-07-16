#!/bin/bash
#PBS -N qe_Si_unfolding
#PBS -l nodes=1:ppn=44
#PBS -l walltime=2:00:00
#PBS -q small
#PBS -o /sfihome/badal.mondal/err/"${PBS_JOBNAME}_${PBS_JOBID}.out"
#PBS -e /sfihome/badal.mondal/err/"${PBS_JOBNAME}_${PBS_JOBID}.err"
#PBS -W x=NACCESSPOLICY:SINGLEJOB

## ====================== Update module path ============================
source /usr/share/Modules/init/bash
module use --append /local/rocks7-temp/modules
module use --append /sfihome/badal.mondal/local/Modulefiles
module purge

## =================== Total number of CPU allocated =====================
echo "Total CPU count = $PBS_NP"

## ========================== Load modules ===============================
module load dftb/condaDFTB
module load qe/7.3

## =================== Set job submit directory ==========================
JOB_DIRECTORY='/sfiwork/badal.mondal/TestUnfolding/tutorials/QuantumEspresso'

for JJ in {Si,};do
	cd ${JOB_DIRECTORY}/${JJ}/reference_without_SOC
	python ${JOB_DIRECTORY}/${JJ}/run_banduppy_hpc.py
done
