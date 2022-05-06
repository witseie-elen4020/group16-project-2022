# To run on multiple nodes on cluster:
#$mpiexec -hostfile /home/shared/machinefile -np 10 python3 -m mpi4py main.py
# To run in wsl:
#$ mpiexec -n 4 python3 -m mpi4py main.py
import numpy as np
from mpi4py import MPI
import sys
import random
import pandas as pd

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
myHostName = MPI.Get_processor_name()  #machine name running the code

def chunk_preprocessing(df_chunk):
   # line below should look something like: df_chunk.drop('time',axis=1,inplace=True)
    df_chunk.drop(df_chunk.iloc[:,[0,1]],axis=1,inplace=True)
    #df_chunk.drop(df_chunk.iloc[:,1],1)
    return df_chunk
def convertToMagnitudes(df_chunk):
#is it time consuming to convert data frames to numpy back and fourth ?
    df_chunk_magnitudes = np.sqrt((df_chunk.iloc[:,0]*df_chunk.iloc[:,0])+ (df_chunk.iloc[:,1]*df_chunk.iloc[:,1])+ (df_chunk.iloc[:,2]*df_chunk.iloc[:,2]))
    return df_chunk_magnitudes

def main():
      if rank==0:
            chunk_mag_list = [] 
            pd.set_option('display.float_format', '{:.10f}'.format)

            startingRow=0
            #on excel it is +2
            EndingRow= int(sys.argv[2])+1
            chunkSize = int((EndingRow-startingRow)/size)
            print(chunkSize)
            fileName = sys.argv[1]
            #input("Enter the csv file's name that contains the data for which you want statistics (gyroscope.csv, gravity.csv, accelerometer.csv):")
            wt1 = MPI.Wtime()
            wt2 = 0.0
            
            chunks = pd.read_csv(fileName, skiprows =lambda x: x not in range(startingRow, EndingRow ),chunksize = 1000000)

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
            
            '''
            data = pd.read_csv(fileName)
            data['magnitude'] = np.sqrt((data['z']*data['z'])+ (data['y']*data['y'])+ (data['x']*data['x']))
            magData = np.array(data['magnitude'])
'''
            l = len(magData)
            print('magData array size:',l)
            '''
            p3 = np.median(magData)
            print ('median using numpy:', p3) 
            p6 = np.quantile(magData,0.25)
            print ('LQ using numpy:', p6) 
            p7 = np.quantile(magData,0.75)
            print ('UQ using numpy:', p7) 
            p8 = min(magData)
            print ('Min using numpy:', p8) 
            p9 = max(magData)
            print ('Max using numpy:', p9) 
            '''
            quotient, rem = divmod(l, size) # count: the size of each sub-task
            wt1 = MPI.Wtime()
            count = [quotient + 1 if pl < rem else quotient for pl in range(size)]
            count = np.array(count)
            displ = [sum(count[:pl]) for pl in range(size)] # displacement: the starting index of each sub-task
            displ = np.array(displ)
            #print('magDataArray: ', magData)
            #print("NumProcesses:",size)
      else:
            magData = None
            count = np.zeros(size, dtype=int)
            displ = None
            wt1 = 0.0
            wt2 = 0.0
      
      comm.Bcast(count, root=0)
      recvbuf = np.zeros(count[rank]) # initialise recvbuf on all processes
      comm.Scatterv([magData, count, displ, MPI.DOUBLE], recvbuf, root=0)
      #print('After Scatterv, process {} has magData:'.format(rank), recvbuf)
      #comm.Barrier()   # Blocks all process until they reach this point  

      l2 = len(recvbuf)
      # Find each stat of the partial arrays, and then gather them to RANK_0
      partialMedian = np.zeros(1)
      partialMedian[0] = np.median(recvbuf)
      partialLQ = np.zeros(1)
      partialLQ[0] = np.quantile(recvbuf,0.25)
      partialUQ = np.zeros(1)
      partialUQ[0] = np.quantile(recvbuf,0.75)
      partialMin = np.zeros(1)
      partialMin[0] = min(recvbuf)
      partialMax = np.zeros(1)
      partialMax[0] = max(recvbuf)
      arrayPartialMedians = np.zeros(size)
      arrayPartialLQ = np.zeros(size)
      arrayPartialUQ = np.zeros(size)
      arrayPartialMin = np.zeros(size)
      arrayPartialMax = np.zeros(size)
      #print('Median for process', rank, 'on node', myHostName, 'is', partialMedian[0]) 
      comm.Gatherv(partialMedian[0], arrayPartialMedians, root=0)
      comm.Gatherv(partialLQ[0],arrayPartialLQ, root=0)
      comm.Gatherv(partialUQ[0],arrayPartialUQ, root=0)
      comm.Gatherv(partialMin[0],arrayPartialMin, root=0)
      comm.Gatherv(partialMax[0],arrayPartialMax, root=0)
      

      #comm.Barrier() 
      if rank == 0:
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
