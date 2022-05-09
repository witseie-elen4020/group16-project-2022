#!/bin/bash

sbatch 'scripts_results/parallel.sh' #99MB and 1 node
sbatch 'scripts_results/parallel1.sh' #99MB and 2 nodes
sbatch 'scripts_results/parallel2.sh' #99MB and 4 nodes
sbatch 'scripts_results/parallel4.sh' #99MB and 10 nodes
sbatch 'scripts_results/parallel8.sh' #987 MB and 1 node
sbatch 'scripts_results/parallel9.sh' #987 MB and 2 nodes
sbatch 'scripts_results/parallel10.sh' #987 MB and 4 nodes
sbatch 'scripts_results/parallel12.sh' #987 MB and 10 nodes
sbatch 'scripts_results/parallel13.sh' #987 MB and 16 nodes
sbatch 'scripts_results/parallel16.sh' #4.9GB and 1 node
sbatch 'scripts_results/parallel17.sh' #4.9GB and 2 nodes
sbatch 'scripts_results/parallel18.sh' #4.9GB and 4 nodes
sbatch 'scripts_results/parallel20.sh' #4.9GB and 10 nodes
sbatch 'scripts_results/parallel21.sh' #4.9GB and 16 nodes
sbatch 'scripts_results/main.sh'  #99MB and 1 node
sbatch 'scripts_results/main2.sh' #99MB and 2 nodes
sbatch 'scripts_results/main3.sh' #99MB and 4 nodes
sbatch 'scripts_results/main5.sh' #99MB and 10 nodes
sbatch 'scripts_results/main6.sh' #99MB and 16 nodes
sbatch 'scripts_results/main10.sh' #987 MB and 1 node
sbatch 'scripts_results/main11.sh' #987 MB and 2 nodes
sbatch 'scripts_results/main12.sh' #987 MB and 4 nodes
sbatch 'scripts_results/main14.sh' #987 MB and 10 nodes
sbatch 'scripts_results/main15.sh' #987 MB and 16 nodes
sbatch 'scripts_results/main19.sh' #4.9GB and 1 node
sbatch 'scripts_results/main20.sh' #4.9GB and 2 nodes
sbatch 'scripts_results/main21.sh' #4.9GB and 4 nodes
sbatch 'scripts_results/main23.sh' #4.9GB and 10 nodes
sbatch 'scripts_results/main24.sh' #4.9GB and 16 nodes
sbatch 'scripts_results/main25.sh' #4.9GB and 32 Processes
sbatch 'scripts_results/main26.sh' #4.9GB and 64 Processes

##sbatch 'scripts_results/main27.sh'    #50GB with 10 nodes
##sbatch 'scripts_results/parallel27.sh' #50GB with 10 nodes