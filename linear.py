# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:55:51 2021

@author: rdaya
"""


import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib.pyplot as plt
#from google.colab import files
import pandas as pd
import streamlit as st
plt.style.use('bmh')


def f2(stock_symbol):
    if stock_symbol=="AAPL":
        df=pd.read_csv('AAPL.csv')
    elif stock_symbol=="GOOG":
        df=pd.read_csv('GOOG.csv')
    elif stock_symbol=="MSFT":
        df=pd.read_csv('MSFT.csv')
    elif stock_symbol=="GME":
        df=pd.read_csv('GME.csv')
    st.write(df.head())
    df= df[['close']]
    forecast_out=30
    df['prediction']= df[['close']].shift(-forecast_out)
    x=np.array(df.drop(['prediction'],1))
    x=x [:-forecast_out]
    y = np.array(df['prediction'])
    y= y[:-forecast_out]
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    lr =LinearRegression()
    lr.fit(x_train,y_train)
    x_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:]
    lr_prediction=lr.predict(x_forecast)
    print(lr_prediction)
    st.write(lr_prediction[29])
    preds = lr.predict(x_test)
    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
    st.sidebar.write("RMSE Value:")
    st.sidebar.write(rms)
    prediction=lr_prediction
    valid=df[x.shape[0]:]
    #valid['prediction'] = 0
    valid['prediction']=prediction
    plt.figure(figsize=(20,10))
    plt.title('Model')
    plt.xlabel('Days')
    plt.ylabel('close')
    plt.plot(df['close'])
    #plt.plot(valid['close','prediction'])
    plt.plot(valid['prediction'])
    plt.plot(valid['close'])
    plt.legend(['orig','val','pred'])
    st.pyplot(plt.show())
    st.write("PLot for next 30 days ")
    