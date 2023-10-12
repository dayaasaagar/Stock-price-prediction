import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import Model
import tensorflow as tf
import pandas_datareader as pdr
from tkinter import *
from tkinter import messagebox
from tkinter import scrolledtext
from numpy import array
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt




def f1(stock_symbol):
	if stock_symbol == 'GME':
		df= pd.read_csv("C:/Users/91845/Downloads/app/GME.csv")
	elif stock_symbol =='GOOG':
		df= pd.read_csv("C:/Users/91845/Downloads/app/GOOG.csv")
	elif stock_symbol =='AAPL':
		df= pd.read_csv("C:/Users/91845/Downloads/app/AAPL.csv")
	elif stock_symbol=='MSFT':
		df=pd.read_csv("C:/Users/91845/Downloads/app/MSFT.csv")
	else:
		st.write("error")
	
	df1=df.reset_index()['close']
	scaler=MinMaxScaler(feature_range=(0,1))
	df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
	training_size=int(len(df1)*0.65)
	test_size=len(df1)-training_size
	train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
	def prepare_train_test(train_data, test_data):
		x_train = []
		y_train = []
		for i in range(60, len(train_data)):
			x_train.append(train_data[i-60:i, 0])
			y_train.append(train_data[i, 0])
		x_test= []
		y_test = []
		for i in range(60, len(test_data)):
			x_test.append(test_data[i-60:i, 0])
			y_test.append(test_data[i, 0])
		x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
		return x_train, y_train, x_test, y_test
	X_train, y_train, X_test, y_test= prepare_train_test(train_data, test_data)
	def get_model():
		model= Sequential()
		model.add(LSTM(units = 200, return_sequences  = True, input_shape = (X_train.shape[1], 1)))
		model.add(LSTM(units = 200, return_sequences = True))
		model.add(LSTM(units = 100, return_sequences = True))
		model.add(LSTM(units = 100))
		model.add(Dense(units = 1))
		model.compile(optimizer = 'adam', loss = 'mean_squared_error')
		return model
	model = get_model()
	model.summary()
	history = model.fit(X_train, y_train, epochs = 50, batch_size = 60)
	st.write("Training Loss Graph")
	st.set_option('deprecation.showPyplotGlobalUse', False)
	def training_loss_graph(history):
		plt.plot(history.history['loss'], label = 'Training  Loss')
		plt.legend()
		plt.xlabel("Epochs")
		plt.ylabel('Loss')
		f=plt.show()
		st.pyplot(f)
	training_loss_graph(history)
	def get_predicted_INV_scaled(X_test):
		predicted_prices = model.predict(X_test)
		predicted_prices = scaler.inverse_transform(predicted_prices)
		prices = scaler.inverse_transform([y_test])
		return prices, predicted_prices
	
	prices, predicted_prices = get_predicted_INV_scaled(X_test)
	st.write("Actual and Predicted Graph")
	def show_graph_result(prices, predicted_prices):
		index = df.index.values[-len(prices[0]):]
		test_result = pd.DataFrame(columns = ['real', 'predicted'])
		test_result['real'] = prices[0]
		test_result['predicted'] = predicted_prices
		test_result.index = index
		test_result.plot(figsize = (16, 10))
		plt.title("Actual and Predicted")
		plt.ylabel("Price")
		plt.xlabel("Date")
		st.pyplot(plt.show())
	show_graph_result(prices, predicted_prices)
	x_input=test_data[340:].reshape(1,-1)
	x_input.shape
	temp_input=list(x_input)
	temp_input=temp_input[0].tolist()
	lst_output=[]
	n_steps=100
	i=0
	while(i<30):
		if(len(temp_input)>100):
			x_input=np.array(temp_input[1:])
			print("{} day input {}".format(i,x_input))
			x_input=x_input.reshape(1,-1)
			x_input = x_input.reshape((1, n_steps, 1))
			yhat = model.predict(x_input, verbose=0)
			print("{} day output {}".format(i,yhat))
			temp_input.extend(yhat[0].tolist())
			temp_input=temp_input[1:]
			lst_output.extend(yhat.tolist())
			i=i+1
		else:
			x_input = x_input.reshape((1, n_steps,1))
			yhat = model.predict(x_input, verbose=0)
			print(yhat[0])
			temp_input.extend(yhat[0].tolist())
			print(len(temp_input))
			lst_output.extend(yhat.tolist())
			i=i+1
	print(lst_output)
	st.write("Forecasting Graph")
	day_new=np.arange(1,101)
	day_pred=np.arange(101,131)
	plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
	plt.plot(day_pred,scaler.inverse_transform(lst_output))
	plt.title("Forecasting Graph")
	plt.xlabel("Days")
	plt.ylabel("Price")
	st.pyplot(plt.show())







































