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
Objective: This script combines all time series *.tsv files generated using 
ExtractingDataForTimeSeriesGeneration.py into *.tsv file for each day for the 
duration of data collection.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Code to combine all the .tsv files into one to get 24 hrs of data
#extension = 'tsv'
x_min = [0,182,591,995,1375,1848,2259,2745,3242,3695,4128,4565,5019,5432,
            5823,6211,6570,7021,7529,8036,8563,9091,9604,10094,10610,11133,
            11653,12152,12637,13005,13365,13721,14068,14410,14764]
x_max = [182,591,995,1375,1848,2259,2745,3242,3695,4128,4565,5019,5432,
            5823,6211,6570,7021,7529,8036,8563,9091,9604,10094,10610,11133,
            11653,12152,12637,13005,13365,13721,14068,14410,14764,15080]
for d in range(len(x_min)):
    lower = x_min[d]
    upper = x_max[d]
    Output = []
    all_filenames = list()
    for i in range(lower,upper):
        all_filenames.append('Output_%d.tsv' %i)
    #all_filenames = [i for i in glob.glob('Output_2115_*.{}'.format(extension))]
    #combine all files in the list
    all_filenames = np.array(all_filenames)
    Output = pd.concat([pd.read_csv(f) for f in all_filenames])
    #export to csv
    Output.to_csv( "Output_day%d.tsv" %(d+1) , index=False, encoding='utf-8-sig')
    