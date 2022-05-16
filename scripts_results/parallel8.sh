#!/bin/bash
#SBATCH --job-name=parallel   #Set the name of the job
#SBATCH --output=scripts_results/results/987MB_1Node_parallel.out  #Path to the job's standard output
#SBATCH --error=scripts_results/results/errFiles/987MB_parallel.err
#SBATCH --time 0:50:00
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node=1 # Request n cores or task per node

# Run it!
exec srun --mpi=pmi2 python3 -m mpi4py src/parrallelimport.py /data/elen4020/project/small/cGULkmfA.csv 0 20000000
