#!/bin/bash
#PBS -N qe_Si_unfolding
#PBS -l nodes=1:ppn=44
#PBS -l walltime=2:00:00
#PBS -q small
#PBS -o "${PBS_JOBNAME}_${PBS_JOBID}.out"
#PBS -e "${PBS_JOBNAME}_${PBS_JOBID}.err"


## =================== Total number of CPU allocated =====================
echo "Total CPU count = $PBS_NP"

## ========================== Load modules ===============================
module purge
module load banduupy
module load codes/QuantumEspresso

## =================== Set job submit directory ==========================
JOB_DIRECTORY='/sfiwork/badal.mondal/TestUnfolding/tutorials/QuantumEspresso'
cd ${JOB_DIRECTORY}/reference_without_SOC
python ${JOB_DIRECTORY}/run_banduppy_qe.py

