import streamlit as st
import app4
import sv
import linear
from PIL import Image
import base64


st.sidebar.header("USER INPUT")

stock_symbol= st.sidebar.selectbox('Select one symbol', [ 'GME', 'GOOG',"AAPL",'MSFT'])
image = Image.open('2.PNG')
st.image(image, caption='Mini Project')
if st.sidebar.button('Predict using LSTM'):
	app4.f1(stock_symbol)
if st.sidebar.button('Predict using SVM'):
	sv.f2(stock_symbol)
if st.sidebar.button('Predict using Linear Regression'):
	linear.f2(stock_symbol)
