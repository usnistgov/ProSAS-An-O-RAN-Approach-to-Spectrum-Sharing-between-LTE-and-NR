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
LTE and NR data for CNN-LSTM Model.
"""
# grid search CNNLSTM 
import pandas as pd
import numpy as np
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(train, config):
    # unpack config
    n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
    n_input = n_seq * n_steps
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    n_features = 1
    train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, n_features))
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None,n_steps,n_features))))
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_nodes, activation='relu'))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model
   
# forecast with a pre-fit model
def model_predict(model, history, config):
	# unpack config
	n_seq, n_steps, _, _, _, _, _ = config
	n_input = n_seq * n_steps
	# prepare data
	x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
	# convert config to a key
	key = config
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	# summarize score
	result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
    n_seq = [1]
    n_steps = [2]
    n_filters = [256] 
    n_kernel = [1]
    n_nodes = [100,150]
    n_epochs = [100,500]
    n_batch = [100,150]
	# create configs
    configs = list()
    for i in n_seq:
        for j in n_steps:
            for k in n_filters:
                for l in n_kernel:
                    for m in n_nodes:
                        for n in n_epochs:
                            for o in n_batch:
                                cfg = [i, j, k, l, m, n, o]
                                configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

# Accuracy metrics
# Mean Absolute Percentage Error (MAPE);Mean Error (ME); 
# Mean Absolute Error (MAE); Mean Percentage Error (MPE); 
# Root Mean Squared Error (RMSE); Lag 1 Autocorrelation of Error (ACF1);
# Correlation between the Actual and the Forecast (corr); 
# Min-Max Error (minmax)
def forecast_accuracy(actual, forecast):
    #mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mape = np.mean((np.abs(actual-forecast)/np.abs(actual))*100) 
    me = np.mean(actual-forecast)             # ME
    mae = mean_absolute_error(actual,forecast)   # MAE
    mpe = np.mean(((actual-forecast)/actual))   # MPE
    rmse = sqrt(mean_squared_error(actual,forecast))  # RMSE
    corr = np.corrcoef(actual,forecast)[0,1]   # corr
    mins = np.amin(np.hstack([actual[:,None], 
                              forecast[:,None]]), axis=1)
    maxs = np.amax(np.hstack([actual[:,None], 
                              forecast[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

file = 'Output_hourly.tsv'
df = pd.read_csv(file,usecols= ['Time','NRB'],sep = '\t',header = 0, 
                 parse_dates=True)
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df.index = df.Time
df = df.drop('Time',axis = 1)
series = df.squeeze()
data = series.values
n_test = int(len(data)*0.34)
# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
	print(cfg, error)
# configuration with minimum rmse
cfg_opt = scores[:1][0][0]
T = series.index
train, test = train_test_split(data, n_test)
train_time, test_time = T[:-n_test], T[-n_test:]
predictions = list()
#fit model
model = model_fit(train, cfg)
#seed history with training dataset
history = [x for x in train]
#step over each time-step in the test set
for i in range(len(test)):
    #fit model and make forecast for history
    yhat = model_predict(model, history, cfg)
    #store forecast in list of predictions
    predictions.append(yhat)
    #add actual observation to history for the next loop
    history.append(test[i])
metrics = forecast_accuracy(test, np.array(predictions).flatten())
print('Optimal Configuration for CNN-LSTM Model:',cfg_opt)
print('Forecast Accuracy for CNN-LSTM Model:',metrics)
# saving the predicted output
d = {'Time':test_time, 'NRB': np.array(predictions).flatten()}
df = pd.DataFrame(data=d)
df.to_csv("CNNLSTM_Model_Prediction_hourly.tsv",index = False, sep = '\t', encoding = 'utf-8-sig')
# Plot chart
plt.figure(figsize=(18, 10), dpi=200)
dtFmt = mdates.DateFormatter('%d-%b') # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt) 
#plt.plot(train_time,train,'-b', label='Average NRBs Allocated (Train)')
plt.plot(test_time,test, '-k', label='Average No. of PRBs Allocated')
plt.plot(test_time,predictions, '-', color='red', label='Predicted Demand')
#plt.title('Average No. of PRBs Allocated with Predictions (Granularity = 1 hour)', fontsize=16)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Average No. of PRBs', fontsize=20)
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.legend(fontsize=20,loc='best')
plt.tight_layout()
plt.savefig('CNNLSTM_Model_GridSearch_1hour.pdf')
plt.show()