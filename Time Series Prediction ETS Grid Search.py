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
LTE and NR data for Exponential Smoothing (ETS) Model.
"""
# grid search ETS
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p = config
	# define model
	history = np.array(history)
	model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
	# fit model
	model_fit = model.fit(optimized=True)#, use_boxcox=b, remove_bias=r)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = exp_smoothing_forecast(history, cfg)
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
				#filterwarnings("ignore")
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
		executor = Parallel(n_jobs=30, backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					cfg = [t,d,s,p]
					models.append(cfg)
	return models

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
    file = 'Output_min_NR.tsv'
    df = pd.read_csv(file,usecols= ['Time','NRB'],sep = '\t',header = 0, 
                     parse_dates=True)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.index = df.Time
    df = df.drop('Time',axis = 1)
    series = df.squeeze()
    data = series.values
    n_test = int(len(data)*0.34)
    # data config
    cfg_list = exp_smoothing_configs()
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    print('done')
    # configuration with minimum rmse
    cfg_opt = scores[:1][0][0]
    T = series.index
    train, test = train_test_split(data, n_test)
    train_time, test_time = T[:-n_test], T[-n_test:]
    err, pred = walk_forward_validation(data, n_test, cfg_opt)
    metrics = forecast_accuracy(test, np.array(pred))
    print('Optimal Configuration for ETS Model:',cfg_opt)
    print('Forecast Accuracy for ETS Model:',metrics)
    # saving the predicted output
    d = {'Time':test_time, 'NRB': pred}
    df = pd.DataFrame(data=d)
    df.to_csv("ETS_Model_Prediction_min_NR.tsv",index = False, sep = '\t', encoding = 'utf-8-sig')
    # Plot chart
    plt.figure(figsize=(18, 7), dpi=200)
    dtFmt = mdates.DateFormatter('%d-%b') # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt) 
    #plt.plot(train_time,train,'-b', label='Average NRBs Allocated (Train)')
    plt.plot(test_time,test, '-k', label='Average No. of PRBs Allocated (Test)')
    plt.plot(test_time,pred, '-', color='red', label='Predicted Demand')
    #plt.title('Average No. of PRBs Allocated with Predictions (Granularity = 1 hour)', fontsize=16)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Average No. of PRBs', fontsize=20)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ETS_Model_GridSearch_min_NR.pdf')
    plt.show()