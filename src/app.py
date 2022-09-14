#####################################
#       Time Series Project         #
#####################################
### Load libraries and modules ###
import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
import os
from prophet import Prophet
from pmdarima.arima import auto_arima
from pylab import rcParams


######################
# Data Preprocessing #
######################
# Loading the datasets
cpu_train_a_raw = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv')
cpu_train_b_raw = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv')
cpu_test_a_raw = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv')
cpu_test_b_raw = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv')

# Create a copy of the original datasets
cpu_train_a = cpu_train_a_raw.copy()
cpu_train_b = cpu_train_b_raw.copy()
cpu_test_a = cpu_test_a_raw.copy()
cpu_test_b = cpu_test_b_raw.copy()

#####################
#    Dataset A      #
#####################
# Convert the dataframe index to a datetime index 
cpu_train_a['datetime'] = pd.to_datetime(cpu_train_a['datetime'])
cpu_train_a = cpu_train_a.set_index('datetime')
cpu_test_a['datetime'] = pd.to_datetime(cpu_test_a['datetime'])
cpu_test_a = cpu_test_a.set_index('datetime')

### Model using ARIMA ###
stepwise_model_a = auto_arima(cpu_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model_a.aic())
stepwise_model_a.fit(cpu_train_a)
future_forecast = stepwise_model_a.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = cpu_test_a.index,columns=['Prediction'])
print('Future forecast dataset A using ARIMA:\n')
print(future_forecast)
# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/arima_ds_a.pkl')
joblib.dump(stepwise_model_a, filename)

#####################
#    Dataset B      #
#####################
# Rename the columns to use prophet
data_train_b = cpu_train_b_raw.copy()
data_train_b.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)
m_b = Prophet()
m_b.fit(data_train_b)
future = m_b.make_future_dataframe(periods=1)
print('Future forecast dataset B using Prophet:\n')
print(future.tail())

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/prophet_ds_b.pkl')
joblib.dump(m_b, filename)