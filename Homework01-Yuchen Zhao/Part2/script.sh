#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --time=00:05:00
#
#SBATCH -p express
#SBATCH -N 1

cd $HOME/csye7374-zhao.yuc/homework1

module load openmpi/3.1.2

./mm_serial

export OMP_NUM_THREADS=8
./mm_omp

mpirun -np 8 -oversubscribe ./mm_mpi