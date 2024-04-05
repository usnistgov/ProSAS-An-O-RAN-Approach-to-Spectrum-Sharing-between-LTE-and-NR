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
Objective: This script resamples the time series data and generate time series data
corresponding to four different granularities: 1 hour, 1 min, 1 sec, and 0.5s.
"""
# Script to resample tsv file and save data
import pandas as pd
import numpy as np

# load dataset
for i in range(1,36):
    file = 'Output_day%d.tsv' % i
    df = pd.read_csv(file,usecols= ['Time','NRB'],sep = '\t',header = 0, 
                     parse_dates=True)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.index = df.Time
    df = df.drop('Time',axis = 1)
    series = df.squeeze()
    
    # upsample data to obtain per hour data
    upsampled_hr = series.resample('H').mean().interpolate(method = 'linear')
    d_hr = {'Time': upsampled_hr.index, 'NRB': upsampled_hr.values}
    df_hourly = pd.DataFrame(data=d_hr)
    
    #export to csv
    df_hourly.to_csv( "Output_day%d_hourly.tsv" %i , index=False, encoding='utf-8-sig', sep ='\t')
    
    # upsample data to obtain per minute data
    upsampled_min = series.resample('1min').mean().interpolate(method = 'linear')
    d_min = {'Time': upsampled_min.index, 'NRB': upsampled_min.values}
    df_min = pd.DataFrame(data=d_min)
    
    #export to csv
    df_min.to_csv( "Output_day%d_min.tsv" %i , index=False, encoding='utf-8-sig', sep ='\t')
    
    # upsample data to obtain per sec data
    upsampled_sec = series.resample('1s').mean().interpolate(method = 'linear')
    d_sec = {'Time': upsampled_sec.index, 'NRB': upsampled_sec.values}
    df_sec = pd.DataFrame(data=d_sec)
    #export to csv
    df_sec.to_csv( "Output_day%d_sec.tsv" %i , index=False, encoding='utf-8-sig', sep ='\t')
    
    # upsample data to obtain per 0.5 sec data
    upsampled_500ms = series.resample('500ms').mean().interpolate(method = 'linear')
    d_500ms = {'Time': upsampled_500ms.index, 'NRB': upsampled_500ms.values}
    df_500ms = pd.DataFrame(data=d_500ms)
    #export to csv
    df_500ms.to_csv( "Output_day%d_500ms.tsv" %i , index=False, encoding='utf-8-sig', sep ='\t')

all_hourly_filenames = list()
all_min_filenames = list()
all_sec_filenames = list()
all_500ms_filenames = list()
for i in range(1,36):
    all_hourly_filenames.append('Output_day%d_hourly.tsv' %i)
    all_min_filenames.append('Output_day%d_min.tsv' %i)
    all_sec_filenames.append('Output_day%d_sec.tsv' %i)
    all_500ms_filenames.append('Output_day%d_500ms.tsv' %i)
#combine all files in the list
all_filenames_hourly = np.array(all_hourly_filenames)
all_filenames_min = np.array(all_min_filenames)
all_filenames_sec = np.array(all_sec_filenames)
all_filenames_500ms = np.array(all_500ms_filenames)
Output_hr = pd.concat([pd.read_csv(f) for f in all_hourly_filenames])
Output_min = pd.concat([pd.read_csv(f) for f in all_min_filenames])
Output_sec = pd.concat([pd.read_csv(f) for f in all_sec_filenames])
Output_500ms = pd.concat([pd.read_csv(f) for f in all_500ms_filenames])
#export to csv
Output_hr.to_csv( "Output_hourly_LTE.tsv", index=False, encoding='utf-8-sig')
Output_min.to_csv( "Output_min_LTE.tsv", index=False, encoding='utf-8-sig')
Output_sec.to_csv( "Output_sec_LTE.tsv", index=False, encoding='utf-8-sig')
Output_500ms.to_csv( "Output_500ms_LTE.tsv", index=False, encoding='utf-8-sig')