"""
Disclaimer: NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium, 
provided that you keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of 
the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed 
the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards 
and Technology as the source of the software. 

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY 
OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, 
AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY 
DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING 
BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated 
with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, 
programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a 
failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection 
within the United States.
"""
"""
Objective: This script generates the time series LTE resource demand data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
from numba import jit, cuda

# code to add missing SFN, Subframe Index and time values.
filename = 'Output_BladeRF_OWL_Modified.tsv'
headerList = ['SFN','Subframe Index','RNTI','Direction','MCS','NRB','TBS',	
              'TBS0','TBS1','DCI message type','NDI0','NDI1','HARQ PID',	
              'NCCE Location','Aggregation Level','CFI','DCI Correctness Check']
tf = pd.read_csv(filename,sep='\t', header=0, chunksize=1e6)
t = 0
for f in tf:
    f.to_csv('Output_BladeRF_OWL_Final.tsv',header = headerList,sep = '\t',index = False)
    df = pd.read_csv('Output_BladeRF_OWL_Final.tsv',sep='\t',header = 0)
    max_sfn = max(df['SFN'])
    df = df[df['DCI message type'] == 8]
    # find the unique values of System Frame Number (SFN) and for each SFN 
    # create the corresponding 0-9 subframe indices
    if t==0:
        val = "2023-01-19 11:33:00.000"
        min_sfn = 0
        sfn = np.repeat(np.arange(min_sfn, max_sfn),10)
        sfn_uniq = np.arange(min_sfn,max_sfn)
    else:
        sfn = np.repeat(np.arange(min_sfn,max_sfn),10)
        sfn_uniq = np.arange(min_sfn,max_sfn)
    sfid = np.tile(np.arange(0,10),len(sfn_uniq))
    new_df = pd.DataFrame(columns=['Time','SFN','Subframe Index','NRB'])
    new_df['SFN'] = sfn
    new_df['Subframe Index'] = sfid
    new_df['Time'] = pd.date_range(start= val,periods = (max_sfn-min_sfn)*10, freq = 'L')
    for n in range(0,len(sfn_uniq)):
        id_df = df[df['SFN']==sfn_uniq[n]].index.values
        id_newdf = new_df[new_df['SFN']==sfn_uniq[n]].index.values
        for i in range(0,len(id_newdf)):
            temp = []
            for j in range(0, len(id_df)):
                if new_df['Subframe Index'][id_newdf[i]] == df['Subframe Index'][id_df[j]]:
                    temp.append(df['NRB'][id_df[j]])
            new_df['NRB'][id_newdf[i]] = sum(temp)
    header = ['Time','SFN','Subframe Index','NRB']
    new_df.to_csv('Output_'+str(t)+'.tsv', columns = header, sep = '\t', index = False)
    t=t+1
    val = new_df.Time.max()
    min_sfn = max_sfn
