import pandas as pd
import numpy as np
from mpi4py import MPI
import sys
import seaborn
import datetime
import os 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()          #number of the process running the code
numProcesses = comm.Get_size()  #total number of processes running
myHostName = MPI.Get_processor_name()  #machine name running  the code
end_time = 0.0 
pd.set_option('display.float_format', '{:.10f}'.format)
 
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
    df_chunk.drop(df_chunk.iloc[:,[0,1]],axis=1,inplace=True)
    return df_chunk

def convertToMagnitudes(df_chunk):
    df_chunk_magnitudes = np.sqrt((df_chunk.iloc[:,0]*df_chunk.iloc[:,0])+ (df_chunk.iloc[:,1]*df_chunk.iloc[:,1])+ (df_chunk.iloc[:,2]*df_chunk.iloc[:,2]))
    return df_chunk_magnitudes

def readInData(start, stop, fileName):
    chunk_mag_list = []
    startingRow=start
    endingRow= stop
    chunkSize = int((endingRow-startingRow)/numProcesses)
    if chunkSize<1:   
        if rank==0:
            print("Exiting as there are not enough rows for processes")
        exit()
      
    if chunkSize>20000000:
        chunkSize = 20000000  #ensures it will fit into Ram
    chunks = pd.read_csv(fileName, skiprows =lambda x: x not in range(startingRow, endingRow ),chunksize=chunkSize)
    totalProcessingTime = 0.0
    i =0 
    start_time = 0.0
    wt4 = 0.0
    wt5 = 0.0
    for df_chunk in chunks:
        if i ==0: #obtaining the start and end time from the UNIX time stamp
            if rank==0:
                start_time = datetime.datetime.utcfromtimestamp((df_chunk.iloc[1,0])/1e9).strftime('%Y-%m-%d %H:%M:%S')     
        end_time = datetime.datetime.utcfromtimestamp((df_chunk.iloc[-1,0])/1e9).strftime('%Y-%m-%d %H:%M:%S')
        df_chunk_filtered = df_chunk
        if rank==0:
            wt4 =MPI.Wtime()
        chunkPreprocessing(df_chunk)
        df_chunk_magnitudes =convertToMagnitudes(df_chunk_filtered)
        if rank==0:
            wt5 =MPI.Wtime()
        chunk_mag_list.append(df_chunk_magnitudes) # Once the magnitudes for a chunk are done, append the magnitutes to list
        timeForRound = wt5-wt4
        totalProcessingTime += timeForRound
        i+=1
 
    df_concat = pd.concat(chunk_mag_list) # concantenate the list of magnitudes into dataframe
    magData=np.array(df_concat) #convert into np array 
    return magData, totalProcessingTime, start_time, end_time 
  
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

def printTimes(wt0Start, wt1Start,wt1End,wt4Start,wt4End,wt0End,totalProcessingTime, start_time, end_time):
      print('Time taken Reading IN (ms): ', (wt1End-wt1Start - totalProcessingTime)*1000)
      print('Time taken for preprocessing, magnitude and numpy conversion(ms): ', (totalProcessingTime)*1000)
      print('Time taken scatter, compute, gather (ms): ', (wt4End-wt4Start)*1000)
      print('Time taken overall (ms): ', (wt0End-wt0Start)*1000)
      print('Start time and End Time (GMT):', start_time, ' , ', end_time)
      
def main():
    
    if rank == 0:        
        wt0Start = MPI.Wtime() #begin timing whole code
    fileName = sys.argv[1]   
    fileSize=os.path.getsize(fileName)
    bytesPerLine=68.2
    maxLines=int((fileSize/bytesPerLine)*0.70) #This is to ensure that the number of lines remains within the file size
    startingRow=int(sys.argv[2])+1
    endingRow= int(sys.argv[3])+1
    if startingRow>maxLines :
        if rank==0:
            print("Starting row invalid")
        exit()   
    if endingRow>maxLines:
        endingRow=maxLines
        if rank == 0:
            print("You requested too many lines therefore stopped at line {}".format(maxLines))
    totalNumRows = (endingRow-startingRow)   
    if totalNumRows%numProcesses!=0:
        if rank==0:
            print("Your row number has been rounded down to ensure all processes share equal work loads to maximise efficiency")
    if (numProcesses < totalNumRows):
        rowsPerProcess = int(totalNumRows / numProcesses)
        start = rank * rowsPerProcess
        stop = start + rowsPerProcess
        # do the work within the range set aside for this process
        if rank == 0:
            wt1Start = MPI.Wtime()   #start reading in time
        x, totalProcessingTime, start_time, end_time = readInData(startingRow+start, startingRow+stop, fileName)
        if rank == 0:
            wt1End = MPI.Wtime()
    else:
        x = 0.0
        if rank == 0 :  # cannot break into equal chunks; one process reports the error
            print("Must be run with more rows than processes")
        exit()
    
    if rank ==0:
            wt4Start = MPI.Wtime()
    partialMedian = np.zeros(1)
    partialMedian[0] = np.median(x)
    partialLQ = np.zeros(1)
    partialLQ[0] = np.quantile(x,0.25)
    partialUQ = np.zeros(1)
    partialUQ[0] = np.quantile(x,0.75)
    partialMin = np.zeros(1)
    partialMin[0] = min(x)
    partialMax = np.zeros(1)
    partialMax[0] = max(x)
    arrayPartialMedians = np.zeros(numProcesses) # array of medians from each chunk 
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
    if rank ==0:
        wt4End = MPI.Wtime() 
        med = np.median(arrayPartialMedians)
        LQ = np.quantile(arrayPartialLQ, 0.25)
        UQ = np.quantile(arrayPartialUQ, 0.75)
        MINreal = min(arrayPartialMin)
        MAXreal = max(arrayPartialMax)
        IQR, Min, Max = outliers(LQ,UQ, MINreal,MAXreal)
        wt0End = MPI.Wtime()
        printResults(med,LQ,UQ,IQR,Max,Min)
        printTimes(wt0Start, wt1Start,wt1End,wt4Start,wt4End,wt0End,totalProcessingTime, start_time, end_time)
        exportBoxAndWhisker(Min, LQ, med, UQ, Max, start_time, end_time, fileName,numProcesses)
    MPI.Finalize()
    exit()

if (__name__ == "__main__"):
    main()