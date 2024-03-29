#!/bin/bash
#SBATCH --job-name=parallel   #Set the name of the job
#SBATCH --output=scripts_results/results/99MB_2Nodes_parallel.out  #Path to the job's standard output
#SBATCH --error=scripts_results/results/errFiles/99MB_parallel.err
#SBATCH --time 0:20:00
#SBATCH --nodes 2
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node=1 # Request n cores or task per node

# Run it!
exec srun --mpi=pmi2 python3 -m mpi4py src/parrallelimport.py /data/elen4020/project/small/9IWgmjUj.csv 0 2000000
