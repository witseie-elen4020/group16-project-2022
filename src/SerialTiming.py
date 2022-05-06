# $mpiexec python3 -m mpi4py SerialTiming.py

import numpy as np
import pandas as pd
from mpi4py import MPI
import sys
import csv
from csv import reader
from csv import DictReader

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

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
      wt1 = MPI.Wtime()

      chunk_mag_list = [] 
      pd.set_option('display.float_format', '{:.10f}'.format)

      #should define size of numpy arrays. Mot efficeint to append and change the size
      #write code to obtain first time entry from the csv and obtainthe time between samples
      #what about the headings. Are tbey not going to cause a problem

      #will be calcualted based on requested period, the first time entry and 1/(sampling rate)
      '''
      startingRow=0
      #on excel it is +2
      EndingRow= int(sys.argv[2])+1
      chunkSize = int((EndingRow-startingRow)/10)
      '''
      # fileName = input("Enter the csv file's name that contains the data for which you want statistics (gyroscope.csv, gravity.csv, accelerometer.csv):")
      fileName = sys.argv[1]


      
      
      wt1 = MPI.Wtime()
      data = pd.read_csv(fileName)
      wt2 = MPI.Wtime()

      wt20 = MPI.Wtime()
      data['magnitude'] = np.sqrt((data['z']*data['z'])+ (data['y']*data['y'])+ (data['x']*data['x']))
      magData = np.array(data['magnitude'])
      wt21 = MPI.Wtime()

      fileLines = len(magData)

      '''
      #ensures it will fit into Ram
      chunks = pd.read_csv(fileName, skiprows =lambda x: x not in range(startingRow, EndingRow ),chunksize=1000000)
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
      
      #wt4 = MPI.Wtime()
      '''
      wt7 = MPI.Wtime()
      med = np.median(magData)
      wt8 = MPI.Wtime()

      wt9 = MPI.Wtime()
      LQ = np.quantile(magData, 0.25)
      wt10 = MPI.Wtime()

      wt11 = MPI.Wtime()
      UQ = np.quantile(magData, 0.75)
      wt12 = MPI.Wtime()

      wt13 = MPI.Wtime()
      MINreal = min(magData)
      IQR = UQ-LQ
      Min = LQ - 1.5*IQR #with exclusion of outliers
      if Min<MINreal:
            Min = MINreal
      wt14 = MPI.Wtime()

      wt15 = MPI.Wtime()
      MAXreal = max(magData)
      Max = UQ + 1.5*IQR
      if Max>MAXreal:
            Max = MAXreal  
      wt16 = MPI.Wtime()
    
      print('Median:', med)
      print('LQ', LQ)
      print('UQ',UQ)
      print('IQR',IQR)
      print('MAX:', Max)
      print('MIN', Min)
      
      print('\nTiming In milliseconds for file size', fileLines, )
      print('Reading in : ', (wt2-wt1)*1000)
      #print('Reading in chunks with magnitude: ', (wt4-wt3)*1000)

      print('Magnitude in one go: ', (wt21-wt20)*1000)
      #print('Magnitude in chunks: ', (wt31-wt30)*1000)
      #print('Magnitude: ', (wt6-wt5)*1000)
      print('Median: ', (wt8-wt7)*1000)
      print('Lower Q: ', (wt10-wt9)*1000)
      print('Upper Q: ', (wt12-wt11)*1000)
      print('Min: ', (wt14-wt13)*1000)
      print('Max: ', (wt16-wt15)*1000)
      
      wt2 = MPI.Wtime()
      print('Time taken (ms): ', (wt2-wt1)*1000)
      MPI.Finalize()
      exit()

if (__name__ == "__main__"):
    main()