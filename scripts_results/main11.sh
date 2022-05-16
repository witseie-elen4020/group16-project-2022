#!/bin/bash
#SBATCH --job-name=main   #Set the name of the job
#SBATCH --output=scripts_results/results/987MB_2Nodes_main.out  #Path to the job's standard output
#SBATCH --error=scripts_results/results/errFiles/987MB_main.err
#SBATCH --time 0:20:00
#SBATCH --nodes 2
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node=1 # Request n cores or task per node

# Run it!
exec srun --mpi=pmi2 python3 -m mpi4py src/main.py /data/elen4020/project/small/cGULkmfA.csv 0 20000000
