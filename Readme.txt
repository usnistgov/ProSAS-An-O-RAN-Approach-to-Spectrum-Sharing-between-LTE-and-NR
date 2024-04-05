***ProSAS: An O-RAN Approach to Spectrum Sharing between NR and LTE (Dataset and Code)***

**Brief Description**
Proactive Spectrum Adaptation Scheme (ProSAS) is a data-driven Open-RAN (O-RAN)-compatible resource-sharing solution. It capitalizes on O-RAN’s capabilities, focusing on intelligent demand prediction 
and resource allocation to minimize resource surpluses or deficits experienced by networks. It utilizes statistical models and deep learning techniques to analyze and predict radio resource demand patterns. 
This analysis uses real-world Long Term Evolution (LTE) resource usage data collected using BladeRF Software Defined Radio (SDR) and Online Watcher of LTE (OWL), an open-source LTE sniffer, as well as 
New Radio (NR) data generated synthetically using Time-Series Generative Adversarial Network (TimeGAN). In this document, we describe the steps followed for data collection and the scripts used for
data processing, demand predictions, and resource allocation.

**Requirements**
Software: Ubuntu 20.04, OWL, Anaconda3, and MATLAB 2023b.
Hardware: Ettus USRP B210 or BladeRF SDR board. 

**Data Collection and Processing**
*We compiled an extensive dataset of LTE scheduling information, collected at the National Institute of Standards and Technology (NIST) Gaithersburg campus between January and February 2023. 
*The data was collected from the downlink traffic at 2115 MHz (Band 4) using a free and open-source LTE control channel decoder called the Online Watcher of LTE (OWL), developed by IMDEA Networks 
Institute.
*For the collection of real-world LTE signals, we used a BladeRF SDR board to transmit the data to a PC running Ubuntu 20.04 and executing OWL to decode the signals. It's worth noting that instead of 
BladeRF, Ettus USRP B210 can also be used. You can find the details for the installation of Ettus USRP B210, BladeRF SDR and OWL in the document "Installation_Manual for OWL.pdf".
*The raw data collected using OWL and BladeRF is available in the file "Output_BladeRF_OWL.tsv". The file includes the following data:
	· System Frame Number (SFN): internal timing of LTE (1 every frame = 10 ms)
	· Subframe Index from 0 to 9 (1 subframe = 1 ms)
	· Radio Network Temporary Identifier (RNTI) in decimal
	· Direction: 1 = downlink; 0 = uplink
	· Modulation and Coding Scheme (MCS) in 0 - 31
	· Number of allocated resource blocks in 0 - 110
	· Transport block size in bits
	· Transport block size in bits (code word 0), -1 if n/a
	· Transport block size in bits (code word 1), -1 if n/a
	· Downlink Control Message (DCI) message type. 
	· New data indicator toggle for codeword 0
	· New data indicator toggle for codeword 1
	· Hybrid Automatic Repeat Request (HARQ) process id
	· Narrowband Control Channel Element (NCCE) location of the DCI message
	· Aggregation level of the DCI message
	· Control Frame Indicator (CFI)
	· DCI correctness check.
*We extracted the System Frame Number (SFN), Subframe Index, and number of allocated resource blocks corresponding to DCI message type 8, which corresponds to the subframes carrying data. 
We filled in the missing data for the missing subframe indices, added the time of data collection, and compiled a time series of the PRB usage for LTE. We also downsampled the data from a granularity 
of 1 ms (per subframe) to 1 hour, 1 minute, 1 second, and 500 milliseconds using the mean function. Run the python scripts listed below in the given order to obtain the data corresponding to the 
mentioned granularities:
	· Transform BladeRF+OWL Data.py
	· Extracting Data For Time Series Generation.py
	· Generating Per Day Data.py
	· Resampling Time Series Data.py
*Using the resampled LTE demand time series data, we generated synthetic NR demand data using TimeGAN. You can run the python script "NR Synthetic Data Generation-TimeGAN.py" to obtain 
synthetic NR data using the collected LTE data. 
*We used the MATLAB script "CDF_LTE_NR_Demand_Plot.m" to generate and compare the Cumulative Distribution Function (CDF) for collected LTE data and synthetically generated NR data. The goal 
is to highlight the similarity in demand distribution between the compiled LTE dataset and the synthetically generated NR dataset, thereby confirming the effectiveness of TimeGAN.

**Demand Prediction**
* We use a toolkit of time-series predictive models that incorporates statistical methods and deep learning architectures. 
	· Statistical Models: Autoregressive Integrated Moving Average (ARIMA) model and Exponential Smoothing (ETS) model.
	· Deep Learning Models: Multilayer Perceptron (MLP) Model, Convolutional Neural Network (CNN) Model, Long Short Term Memory (LSTM) Model, CNN-LSTM, and Convolutional LSTM (ConvLSTM).
	· Baseline Models: Persistence or Naive model, Moving Average (MA) model, and Moving Median (MM) model.  
* The implementation of these prediction models is carried out in Python, with Keras serving as the backend for deep learning architectures. For each model, we have designed a framework for grid-searching
model hyperparameters, employing RMSE as our guiding metric through one-step walk-forward validation.  For training and validating the model architecture, we use 66 % for training at each granularity, i.e., 
1-hour and 1-minute. Lastly, for the deep learning architectures, we use the Adam optimization algorithm and employ Rectified Linear Unit (ReLU) as the activation function.
* For each of the models, we have created scripts to execute the grid search. These scripts include:
	· Time Series Prediction Baseline Grid Search.py (Persistence or Naive Model + Moving Average (MA) Model + Moving Median (MM) Model)
	· Time Series Prediction ARIMA Grid Search.py
	· Time Series Prediction ETS Grid Search.py
	· Time Series Prediction MLP Grid Search.py
	· Time Series Prediction CNN Grid Search.py
	· Time Series Prediction LSTM Grid Search.py
	· Time Series Prediction CNNLSTM Grid Search.py
	· Time Series Prediction ConvLSTM Grid Search.py
* To compare the prediction models use the MATLAB script "Model_Comparison_Plot_Generation.m".

**Resource Allocation**
* Resource allocation is the final step where we allocate resources based on predicted demand, aiming to minimize the surplus or deficit seen by the networks. Use the MATLAB script 
"Resource_Allocation_Scheme.m" to get optimal resource allocation.

***If you have any queries, please feel free to reach out to Sneihil Gopal at sneihil.gopal@nist.gov.***

References:
[1] Brownlee, Jason. "Introduction to time series forecasting with python." Jason Brownlee (2019).
[2] Brownlee, Jason. "Deep Learning for time series forecasting." Jason Brownlee (2020).
[3] Brownlee, Jason. Long short-term memory networks with python: develop sequence prediction models with deep learning. Machine Learning Mastery, 2017.

