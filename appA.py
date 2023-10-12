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

root=Tk()
root.title("Stock Price Prediction")
root.geometry("500x600+100+200")

def f1():
	key="813f03ac5dfc87e23332293462b40c099460e615"
	comp=''
	choice=covar.get()
	if choice=='AAPL':
		comp='AAPL'
	if choice=='MSFT':
		comp='MSFT'
	if choice=='GOOG':
		comp='GOOG'
	if choice=='GME':
		comp='GME'
	df = pdr.get_data_tiingo(comp, api_key=key)
	df.to_csv('GOOG.csv')
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
	history = model.fit(X_train, y_train, epochs = 100, batch_size = 60)
	def training_loss_graph(history):
		plt.plot(history.history['loss'], label = 'Training  Loss')
		plt.legend()
		plt.xlabel("Epochs")
		plt.ylabel('Loss')
		plt.show()
	training_loss_graph(history)
	def get_predicted_INV_scaled(X_test):
		predicted_prices = model.predict(X_test)
		predicted_prices = scaler.inverse_transform(predicted_prices)
		prices = scaler.inverse_transform([y_test])
		return prices, predicted_prices
	prices, predicted_prices = get_predicted_INV_scaled(X_test)
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
		plt.show()
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
	day_new=np.arange(1,101)
	day_pred=np.arange(101,131)
	plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
	plt.plot(day_pred,scaler.inverse_transform(lst_output))
	plt.show()
		

	
    	

lbl=Label(root,text="Stock Price Prediction",font=('comic sans ms',16,'bold'))

covar=StringVar(root)
choices={'AAPL','GOOG','MSFT','GME'}
covar.set('AAPL')
popupMenu=OptionMenu(root,covar,*choices)
api=Label(root,text="Take Data using API",font=('comic sans ms',16,'bold'))

def change_dropdown(*args):
	print(covar.get())
covar.trace('w',change_dropdown)


predict=Button(root,text="Predict",font=('comic sans ms',16,'bold'),command=f1)
lbl.pack(pady=10)
api.pack(pady=10)
popupMenu.pack(pady=10)
predict.pack(pady=10)

root.mainloop()
