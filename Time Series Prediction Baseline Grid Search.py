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
Objective: Script to perform grid search for hyperparameters corresponding to 
LTE and NR data for Persistence or Naive Model, Moving Average (MA) and 
Moving Median (MM) Model.
"""
# grid search simple forecast
import pandas as pd
import numpy as np
import itertools
import timeit
import matplotlib.pyplot as plt
from scipy import stats
from math import pi, sqrt, ceil
from warnings import catch_warnings
from warnings import filterwarnings

from math import sqrt
from numpy import mean
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, mean_absolute_error
from pandas import read_csv

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error, predictions

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = cfg
	# show all warnings and fail on exception if debugging
	if debug:
		result, pred = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result, pred = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, t, offsets=[1]):
	configs = list()
	#for t in ['persist', 'mean', 'median']:
	for o in offsets:
		for i in range(1, max_length+1):
			cfg = [i, o, t]
			configs.append(cfg)
	return configs

# Accuracy metrics
# Mean Absolute Percentage Error (MAPE);Mean Error (ME); 
# Mean Absolute Error (MAE); Mean Percentage Error (MPE); 
# Root Mean Squared Error (RMSE); Lag 1 Autocorrelation of Error (ACF1);
# Correlation between the Actual and the Forecast (corr); 
# Min-Max Error (minmax)
def forecast_accuracy(actual, forecast):
    mape = np.mean((np.abs(actual-forecast)/np.abs(actual))*100) 
    me = np.mean(actual-forecast)             
    mae = mean_absolute_error(actual,forecast)   
    mpe = np.mean(((actual-forecast)/actual))   
    rmse = sqrt(mean_squared_error(actual,forecast))  
    corr = np.corrcoef(actual,forecast)[0,1]   
    mins = np.amin(np.hstack([actual[:,None], 
                              forecast[:,None]]), axis=1)
    maxs = np.amax(np.hstack([actual[:,None], 
                              forecast[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

if __name__ == '__main__':
    # load dataset
    file = 'Output_hourly.tsv'
    df = pd.read_csv(file,usecols= ['Time','NRB'],sep = '\t',header = 0, 
                     parse_dates=True)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.index = df.Time
    df = df.drop('Time',axis = 1)
    series = df.squeeze()
    data = series.values
    n_test = int(len(data)*0.34)
    # data config
    max_length = 10#len(data) - n_test
    cfg_opt = list()
    for t in ['persist', 'mean', 'median']:
        cfg_list = simple_configs(max_length,t)
        # grid search
        scores = grid_search(data, cfg_list, n_test)
        cfg_opt.append(min(scores))
        print('done')
    # compute predictions for optimal configuration of each baseline
    i = 0
    T = series.index
    train, test = train_test_split(data, n_test)
    train_time, test_time = T[:-n_test], T[-n_test:]
    for cfg in cfg_opt:
        err, pred = walk_forward_validation(data, n_test, cfg[0])
        if i==0:
            persist_err = err
            persist_pred = pred
            persist_metrics = forecast_accuracy(test, np.array(persist_pred))
            print('Optimal Configuration for Persistence Model:',cfg[0])
            print('Forecast Accuracy for Persistence Model:',persist_metrics)
            # saving the predicted output
            d = {'Time':test_time, 'NRB': persist_pred}
            persist_df = pd.DataFrame(data=d)
            persist_df.to_csv("Persistence_Model_Prediction_hourly.tsv",index = False, sep = '\t', encoding = 'utf-8-sig')
            # Plot chart
            plt.figure(figsize=(18, 10), dpi=200)
            dtFmt = mdates.DateFormatter('%d-%b') # define the formatting
            plt.gca().xaxis.set_major_formatter(dtFmt) 
            #plt.plot(train_time,train,'-b', label='Average No. of PRBs Allocated (Train)')
            plt.plot(test_time,test, '-k', label='Average No. of PRBs Allocated')
            plt.plot(test_time,persist_pred, '-', color='red', label='Predicted Demand')
            #plt.title('Average No. of PRBs Allocated by Persistence Model (Granularity = 1 hour)', fontsize=16)
            plt.xlabel('Time', fontsize=45)
            plt.ylabel('Average No. of PRBs', fontsize=45)
            plt.xticks(fontsize=45)
            plt.yticks(fontsize=45)
            plt.legend(fontsize=30, loc='best')
            plt.tight_layout()
            plt.savefig('Persistence_Model_GridSearch_hourly.pdf')
            plt.show()
        elif i==1:
            mean_err = err
            mean_pred = pred
            mean_metrics = forecast_accuracy(test, np.array(mean_pred))
            print('Optimal Configuration for Average Model:',cfg[0])
            print('Forecast Accuracy for Average Model:',mean_metrics)
            # saving the predicted output
            d = {'Time':test_time, 'NRB': mean_pred}
            mean_df = pd.DataFrame(data=d)
            mean_df.to_csv("MA_Model_Prediction_hourly.tsv",index = False, sep = '\t', encoding = 'utf-8-sig')
            # Plot chart
            plt.figure(figsize=(18, 10), dpi=200)
            dtFmt = mdates.DateFormatter('%d-%b') # define the formatting
            plt.gca().xaxis.set_major_formatter(dtFmt) 
            #plt.plot(train_time,train,'-b', label='Average No. of PRBs Allocated (Train)')
            plt.plot(test_time,test, '-k', label='Average No. of PRBs Allocated')
            plt.plot(test_time,mean_pred, '-', color='green', label='Predicted PRB Demand')
            #plt.title('Average No. of PRBs Allocated by MA Model (Granularity = 1 hour)', fontsize=16)
            plt.xlabel('Time', fontsize=45)
            plt.ylabel('Average No. of PRBs', fontsize=45)
            plt.xticks(fontsize=45)
            plt.yticks(fontsize=45)
            plt.legend(fontsize=30, loc='best')
            plt.tight_layout()
            plt.savefig('Mean_Model_GridSearch_hourly.pdf')
            plt.show()
        else:
            median_err = err
            median_pred = pred
            median_metrics = forecast_accuracy(test, np.array(median_pred))
            print('Optimal Configuration for Median Model:',cfg[0])
            print('Forecast Accuracy for Median Model:',median_metrics)
            # saving the predicted output
            d = {'Time':test_time, 'NRB': median_pred}
            median_df = pd.DataFrame(data=d)
            median_df.to_csv("MM_Model_Prediction_hourly.tsv",index = False, sep = '\t', encoding = 'utf-8-sig')
            # Plot chart
            plt.figure(figsize=(18, 10), dpi=200)
            dtFmt = mdates.DateFormatter('%d-%b') # define the formatting
            plt.gca().xaxis.set_major_formatter(dtFmt) 
            #plt.plot(train_time,train,'-b', label='Average No. of PRBs Allocated (Train)')
            plt.plot(test_time,test, '-k', label='Average No. of PRBs Allocated')
            plt.plot(test_time,median_pred, '-', color='purple', label='Predicted Demand')
            #plt.title('Average No. of PRBs Allocated with MM Model (Granularity = 1 hour)', fontsize=16)
            plt.xlabel('Time', fontsize=45)
            plt.ylabel('Average No. of PRBs', fontsize=45)
            plt.xticks(fontsize=45)
            plt.yticks(fontsize=45)
            plt.legend(fontsize=30, loc='best')
            plt.tight_layout()
            plt.savefig('Median_Model_GridSearch_hourly.pdf')
            plt.show()
        i+=1
    # Plot chart
    plt.figure(figsize=(18, 10), dpi=200)
    dtFmt = mdates.DateFormatter('%d-%b') # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt) 
    #plt.plot(train_time,train,'-b', label='Average NRBs Allocated (Train)')
    plt.plot(test_time,test, '-k', label='Average No. of PRBs Allocated')
    plt.plot(test_time,persist_pred, '-', color='red', label='Predicted Demand (Persistence Model)')
    plt.plot(test_time,mean_pred, '-', color='green', label='Predicted Demand (MA Model)')
    plt.plot(test_time,median_pred, '-', color='purple', label='Predicted Demand (MM Model)')
    #plt.title('Average No. of PRBs Allocated using Baseline Models (Granularity = 1 hour)', fontsize=16)
    plt.xlabel('Time', fontsize=45)
    plt.ylabel('Average No. of PRBs', fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.legend(fontsize=30, loc='best')
    plt.tight_layout()
    plt.savefig('NaiveModelComparison_GridSearch_hourly.pdf')
    plt.show()

    # rmse plot
    errors = [persist_err, mean_err, median_err]
    plt.figure(figsize=(18, 10), dpi=200)
    plt.bar(['Persistence Model','MA Model','MM Model'],errors)
    plt.title('Baseline Model Comparison (Granularity = 1 hour)',fontsize=16)
    plt.xlabel('Baseline Model', fontsize=45)
    plt.ylabel('RMSE', fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.legend(fontsize=30, loc='best')
    plt.tight_layout()
    plt.savefig('BarPlotNaiveModelComparison_GridSearch_hourly.pdf')
    plt.show()
