#!/bin/bash
#SBATCH --job-name=parallel   #Set the name of the job
#SBATCH --output=scripts_results/results/4.9GB_10Nodes_parallel.out  #Path to the job's standard output
#SBATCH --error=scripts_results/results/errFiles/4.9GB_parallel.err
#SBATCH --time 1:59:00
#SBATCH --nodes 10
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node=1 # Request n cores or task per node

# Run it!
exec srun --mpi=pmi2 python3 -m mpi4py src/parrallelimport.py /data/elen4020/project/small/LEzPanWg.csv 0 100000000
