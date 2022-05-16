# group16-project-2022
### ELEN4020A
#### Authors: Benjamin Palay(1815593), Gia Croock(2128541), Rael Ware(2153459) }

There is an option of slurm batches as well as manual execution. 
This source code must be run on the wits.eie jaguar compute cluster for slurm batches and when using different nodes. 

##### There are two src codes: Implementation A (called main.py) and Implementation B (called parrallelimport.py).

Ensure that mpi4py,pandas,seaborn and numpy are installed on each node (jaguar3 to jaguar18). For the latter 3 modules, to install: $pip3 install module_name

From the group16-project-2022 directory, run:

## $ sbatch RunAll.sh

This will schedule several batch files for execution into a queue. $squeue to see the queue. 
Each batch file executes code for different sized files and different numbers of processes for each code implementation. The results will be found in scripts_results/results and produced error files in scripts_results/results/errFiles . Each iteration produces its own file with the calculated quantiles, minimum and maximum, as well as times taken for the code to run for each section of the code. It also produces a .png labeled as the file name that was processed. This png contains a boxplot as well as the time ranges requested from the UNIX stamps. This is also stored in scripts_results. To view the images, you need to have an editor that can do so, else you will see binary code. 

The batch files import csv files from the /data/elen4020/project directory. 99MB, 987MB and 4.9GB files are run for each code implementation, with iterations of 1,2,4,10 and 16 nodes each. To run all of these files may take several minutes. To run fewer of them, edit the RunAll.sh file and comment the batch file you do not want executed. Larger file sizes were tested but not included as it takes relatively long to run. 

In this way mulitple files can be tested at once, as long as the number of overall nodes requested does not exceed 16.

To run your own iteration of file and number of processes, either edit one of the bacth files, or use:

For main.py:

 ### $ mpiexec -hostfile /home/shared/machinefile -np 10 python3 -m mpi4py src/main.py arg1 arg2 arg3

where 10 is the number of nodes you wish to run on (this can be changed). arg1 is the complete file path to the csv file you want processed, arg2 is the row you want to start from and arg3 is the row you want to end at. For a 1GB file, the number of lines is about 15000000. The number of lines and file size are proportional. You can enter a larger number than the file has and it will process the whole file, but if you want a porton of the file, enter a lower number of lines. 

This will output to the console: the calculated quantiles, minimum and maximum, as well as times taken for the code to run for each section of the code.

It also produces a .png labeled as the file name that was processed. This png contains a boxplot as well as the time ranges from the UNIX stamps based on the number of lines requested.

For parrallelimport.py:

 ### $ mpiexec -hostfile /home/shared/machinefile -np 10 python3 -m mpi4py src/parrallelimport.py arg1 arg2 arg3

 where the arguments are the same as explained for main.py

 To run without the hostfile, simply run:

For main.py:
 ### $ mpiexec -n 4 python3 -m mpi4py src/main.py arg1 arg2
 
For parrallelimport.py:
 ### $ mpiexec -n 4 python3 -m mpi4py src/parrallelimport.py arg1 arg2 arg3
 
 #### Group member contributions:
 Rael: Implementation B, analysis, report writing. 
 Benjamin: Implementation A, testing, report writing. 
 Gia: assisted with both implementations, results plotting and analysis, report writing.
