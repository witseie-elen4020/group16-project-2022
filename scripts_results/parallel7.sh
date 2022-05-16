#!/bin/bash
#SBATCH --job-name=parallel   #Set the name of the job
#SBATCH --output=scripts_results/results/99MB_Processes64_Nodes16_parallel.out  #Path to the job's standard output
#SBATCH --error=scripts_results/results/errFiles/99MB_parallel.err
#SBATCH --time 0:30:00
#SBATCH --nodes 16
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node=4 # Request n cores or task per node

# Run it!
exec srun --mpi=pmi2 python3 -m mpi4py src/parrallelimport.py /data/elen4020/project/small/9IWgmjUj.csv 0 2000000
