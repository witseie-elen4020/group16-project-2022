# To run on multiple nodes on cluster:
#$mpiexec -hostfile /home/shared/machinefile -np 10 python3 -m mpi4py main.py
# To run in wsl:
#$ mpiexec -n 4 python3 -m mpi4py main.py
import numpy as np
from mpi4py import MPI
import sys
import random
import pandas as pd
import seaborn
import datetime


comm = MPI.COMM_WORLD
numProcesses = comm.Get_size() 
rank = comm.Get_rank()
myHostName = MPI.Get_processor_name()  

def exportBoxAndWhisker(Min, q1, q2, q3, Max, starting, ending, fileName, numProcesses):
    stats = [Min, q1, q2, q3, Max]
    bw = seaborn.boxplot(data=stats, orient="h")
    newFileName = fileName.replace(".csv", "") 
    x = newFileName.rfind("/") 
    if len(newFileName)>x:
          newFileName = newFileName[0:0:] + newFileName[x+1::]
    bw.set(title="Box Plot between  {} and {} \n with {} processes for {}.csv".format(starting, ending, numProcesses, newFileName))     
    bw.get_figure().savefig('scripts_results/{}.png'.format(newFileName))
    
def chunkPreprocessing(df_chunk):
    df_chunk.drop(df_chunk.iloc[:,[0,1]],axis=1,inplace=True) #drop time columns
    return df_chunk

def convertToMagnitudes(df_chunk):
    df_chunk_magnitudes = np.sqrt((df_chunk.iloc[:,0]*df_chunk.iloc[:,0])+ (df_chunk.iloc[:,1]*df_chunk.iloc[:,1])+ (df_chunk.iloc[:,2]*df_chunk.iloc[:,2]))
    return df_chunk_magnitudes

def outliers(LQ,UQ, MINreal,MAXreal):
      IQR = UQ-LQ
      Min = LQ - 1.5*IQR #with exclusion of outliers
      if Min<MINreal:
            Min = MINreal
      Max = UQ + 1.5*IQR
      if Max>MAXreal:
            Max = MAXreal  
      return IQR, Min, Max  

def printResults(med,LQ,UQ,IQR,Max,Min):
      print('Median:', med)
      print('LQ', LQ)
      print('UQ',UQ)
      print('IQR',IQR)
      print('MAX:', Max)
      print('MIN', Min)

def printTimes(wt0Start, wt1Start,wt1End,wt4Start,wt4End,wt0End,totalProcessingTime, NumpyTime, start_time, end_time):
      print('Time taken Reading IN (ms): ', (wt1End-wt1Start - totalProcessingTime)*1000)
      print('Time taken for preprocessing, magnitude and numpy conversion(ms): ', (totalProcessingTime + NumpyTime)*1000)
      print('Time taken scatter, compute, gather (ms): ', (wt4End-wt4Start)*1000)
      print('Time taken overall (ms): ', (wt0End-wt0Start)*1000)
      print('Start time and End Time (GMT):', start_time, ' , ', end_time)   
      
def main():
      if rank==0:
            wt0Start = MPI.Wtime() #begin timing whole code
            chunkMagList = [] #to store the concantenated df_chunks
            pd.set_option('display.float_format', '{:.10f}'.format)
            fileName = sys.argv[1]
            startingRow=0
            endingRow= int(sys.argv[2])+1 
            print("This is the ending row{}".format(endingRow))
            chunkSize = int((endingRow-startingRow)/numProcesses)
            if chunkSize>20000000: # in order to ensure it fits in RAM
                  chunkSize = 20000000
            wt1Start = MPI.Wtime()   #start reading in time  
            chunks = pd.read_csv(fileName, skiprows =lambda x: x not in range(startingRow, endingRow ),chunksize = chunkSize)
            i =0 
            start_time = 0.0
            end_time = 0.0
            totalProcessingTime = 0.0
            for df_chunk in chunks:  
                  if i ==0: #obtaining the start and end time from the UNIX time stamp
                        start_time = datetime.datetime.utcfromtimestamp((df_chunk.iloc[1,0])/1e9).strftime('%Y-%m-%d %H:%M:%S')
                  end_time = datetime.datetime.utcfromtimestamp((df_chunk.iloc[-1,0])/1e9).strftime('%Y-%m-%d %H:%M:%S')
                  df_chunk_filtered = df_chunk
                  wt7 = MPI.Wtime()
                  chunkPreprocessing(df_chunk)
                  df_chunk_magnitudes =convertToMagnitudes(df_chunk_filtered)
                  wt8 = MPI.Wtime()
                  timeForRound = wt8-wt7
                  totalProcessingTime += timeForRound
                  i+=1
                  chunkMagList.append(df_chunk_magnitudes) # Once the magnitudes for a chunk are done, append the magnitutes to list  
            df_concat = pd.concat(chunkMagList) # concantenate the list of magnitudes into dataframe
            wt1End = MPI.Wtime()
            wtNumpyStart = MPI.Wtime()
            magData=np.array(df_concat) #convert into np array to scatter
            wtNumpyStop = MPI.Wtime()
            NumpyTime = wtNumpyStop - wtNumpyStart
            l = len(magData)
            quotient, rem = divmod(l, numProcesses) # count: the numProcesses of each sub-task
            count = [quotient + 1 if pl < rem else quotient for pl in range(numProcesses)]
            count = np.array(count)
            displ = [sum(count[:pl]) for pl in range(numProcesses)] # displacement: the starting index of each sub-task
            displ = np.array(displ)
      else:
            magData = None
            count = np.zeros(numProcesses, dtype=int)
            displ = None
      
      if rank ==0:
            wt4Start = MPI.Wtime() #begin timing for scatter,stats computation,gather
      comm.Bcast(count, root=0)
      allocatedData = np.zeros(count[rank]) # initialise allocatedData array on every process
      comm.Scatterv([magData, count, displ, MPI.DOUBLE], allocatedData, root=0)
      l2 = len(allocatedData)
      # Find each stat of the partial arrays, and then gather them to rank0
      partialMedian = np.zeros(1)
      partialMedian[0] = np.median(allocatedData)
      partialLQ = np.zeros(1)
      partialLQ[0] = np.quantile(allocatedData,0.25)
      partialUQ = np.zeros(1)
      partialUQ[0] = np.quantile(allocatedData,0.75)
      partialMin = np.zeros(1)
      partialMin[0] = min(allocatedData)
      partialMax = np.zeros(1)
      partialMax[0] = max(allocatedData)
      arrayPartialMedians = np.zeros(numProcesses)
      arrayPartialLQ = np.zeros(numProcesses)
      arrayPartialUQ = np.zeros(numProcesses)
      arrayPartialMin = np.zeros(numProcesses)
      arrayPartialMax = np.zeros(numProcesses)
      #print('Median for process', rank, 'on node', myHostName, 'is', partialMedian[0]) 
      comm.Gatherv(partialMedian[0], arrayPartialMedians, root=0)
      comm.Gatherv(partialLQ[0],arrayPartialLQ, root=0)
      comm.Gatherv(partialUQ[0],arrayPartialUQ, root=0)
      comm.Gatherv(partialMin[0],arrayPartialMin, root=0)
      comm.Gatherv(partialMax[0],arrayPartialMax, root=0)

      if rank == 0:
            wt4End = MPI.Wtime() 
            med = np.median(arrayPartialMedians)
            LQ = np.quantile(arrayPartialLQ, 0.25)
            UQ = np.quantile(arrayPartialUQ, 0.75)
            MINreal = min(arrayPartialMin)
            MAXreal = max(arrayPartialMax)
            IQR, Min, Max = outliers(LQ,UQ, MINreal,MAXreal)
            wt0End = MPI.Wtime()
            printResults(med,LQ,UQ,IQR,Max,Min)
            printTimes(wt0Start, wt1Start,wt1End,wt4Start,wt4End,wt0End,totalProcessingTime, NumpyTime, start_time, end_time)
            exportBoxAndWhisker(Min, LQ, med, UQ, Max, start_time, end_time, fileName,numProcesses)
      MPI.Finalize()
      exit()

if (__name__ == "__main__"):
    main()
