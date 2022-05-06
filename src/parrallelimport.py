import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
id = comm.Get_rank()            #number of the process running the code
numProcesses = comm.Get_size()  #total number of processes running
myHostName = MPI.Get_processor_name()  #machine name running the code
        
def chunk_preprocessing(df_chunk):
    df_chunk.drop(df_chunk.iloc[:,[0,1]],axis=1,inplace=True)
    return df_chunk
def convertToMagnitudes(df_chunk):
    df_chunk_magnitudes = np.sqrt((df_chunk.iloc[:,0]*df_chunk.iloc[:,0])+ (df_chunk.iloc[:,1]*df_chunk.iloc[:,1])+ (df_chunk.iloc[:,2]*df_chunk.iloc[:,2]))
    return df_chunk_magnitudes


def readInData(start, stop, fileName):
    chunk_mag_list = []
    startingRow=start
#on excel it is +2
    EndingRow= stop
    chunkSize = int((EndingRow-startingRow)/numProcesses)
    #if chunkSize>50000000 or chunkSize<1000000: #keep within RAM size
    chunkSize = 1000000
    #ensures it will fit into Ram
    chunks = pd.read_csv(fileName, skiprows =lambda x: x not in range(startingRow, EndingRow ),chunksize=chunkSize)
    
    for df_chunk in chunks:
        df_chunk_filtered = df_chunk
        chunk_preprocessing(df_chunk)# remove time column
        df_chunk_magnitudes =convertToMagnitudes(df_chunk_filtered)


        # Once the magnitudes for a chunk are done, append the magnitutes to list
        chunk_mag_list.append(df_chunk_magnitudes)

    # concat the list of magnitudes into dataframe 
    df_concat = pd.concat(chunk_mag_list)
    #convert into np array 
    magData=np.array(df_concat)
    return magData
  
    

    
def main():
    
    if numProcesses >= 1 :
       # checkInput(id)

        if id == 0:        # master
            #master: get the command line argument
            #fileName = input("Enter the csv file's name that contains the data for which you want statistics (gyroscope.csv, gravity.csv, accelerometer.csv):")
            fileName = sys.argv[1]
            wt1 = MPI.Wtime() 
            wt2=0.0
        else :
            # worker: start with empty data
            fileName = 'No data'
            wt1 = 0.0
            wt2=0.0
        #initiate and complete the broadcast
        fileName = comm.bcast(fileName, root=0)

    else :
        print("Please run this program with the number of processes \
greater than 1")
        
    #on excel it is +2
    startingRow=0
    EndingRow= int(sys.argv[2])+1
    chunkSize = int((EndingRow-startingRow)/numProcesses)
    #if chunkSize>50000000 or chunkSize<1000000: #keep within RAM size
    chunkSize = 1000000
    REPS = chunkSize

    if ((REPS % numProcesses) == 0 and numProcesses <= REPS):
        # How much of the loop should a process work on?
        chunkSize = int(REPS / numProcesses)
        start = id * chunkSize
        stop = start + chunkSize
        # do the work within the range set aside for this process
        x = readInData(startingRow+start, startingRow + stop, fileName)
    else:
        x = 0.0
        # cannot break into equal chunks; one process reports the error
        if id == 0 :
            print("Please run with number of processes divisible by \
and less than or equal to {}.".format(REPS))
    
    #print('process {} has magData:'.format(id), x)
    partialMedian = np.zeros(1)
    partialMedian[0] = np.median(x)
    #print('process {} has meidan:'.format(id), partialMedian[0])
    partialLQ = np.zeros(1)
    partialLQ[0] = np.quantile(x,0.25)
    partialUQ = np.zeros(1)
    partialUQ[0] = np.quantile(x,0.75)
    partialMin = np.zeros(1)
    partialMin[0] = min(x)
    partialMax = np.zeros(1)
    partialMax[0] = max(x)
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
    
    if id ==0:
        med = np.median(arrayPartialMedians)
        LQ = np.quantile(arrayPartialLQ, 0.25)
        UQ = np.quantile(arrayPartialUQ, 0.75)
        MINreal = min(arrayPartialMin)
        IQR = UQ-LQ
        Min = LQ - 1.5*IQR #with exclusion of outliers
        if Min<MINreal:
                Min = MINreal
        MAXreal = max(arrayPartialMax)
        Max = UQ + 1.5*IQR
        if Max>MAXreal:
                Max = MAXreal  

        print('Median:', med)
        print('LQ', LQ)
        print('UQ',UQ)
        print('IQR',IQR)
        print('MAX:', Max)
        print('MIN', Min)
        wt2 = MPI.Wtime()
        print('Time taken (ms): ', (wt2-wt1)*1000)
    MPI.Finalize()
    exit()

if (__name__ == "__main__"):
    main()