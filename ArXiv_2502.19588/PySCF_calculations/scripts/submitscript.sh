#!/bin/bash
#
#PBS -N tPA_py
#PBS -m a
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=16
#PBS -l mem=50GB
#PBS -A starting_2024_121
#

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4


#For doduo cluster
module purge
module load PySCF
module load matplotlib/3.7.2-gfbf-2023a

cd $STARTDIR
echo "PBS: $PBS_ID"

ls

echo "loaded modules : " `module list` > out.dat

echo "Job started at : "`date` >> out.dat
python tPA1.py >> out.dat
echo "Job ended at : "`date` >> out.dat
