import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Set page title and favicon
st.set_page_config(page_title="Crypto Price Prediction", page_icon=":money_with_wings:")

# Load Model 
model = load_model(r'C:\Users\vijay A\Desktop\BITCOIN\Bitcoin_Price_prediction_Model.keras')

# Page Title
st.title('Cryptocurrency Price Prediction Using Machine Learning')

# Subheader for Bitcoin Price Data
st.subheader('Bitcoin Price Data')
data = pd.DataFrame(yf.download('BTC-USD','2015-01-01','2024-01-01'))
data = data.reset_index()
st.write(data)

# Bitcoin Line Chart
st.subheader('Bitcoin Line Chart')
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
st.line_chart(data)

# Train-test split
train_data = data[:-100]
test_data = data[-200:]

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

# Prepare data for prediction
base_days = 100
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i - base_days:i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Predicted vs Original Prices
st.subheader('Predicted vs Original Prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y.reshape(-1, 1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)
st.write(chart_data)

# Predicted vs Original Prices Chart
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(chart_data)

# Future Bitcoin Price Prediction
m = y
z = []
future_days = 5
for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1, 1)
    inter = [m[-base_days:, 0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    m = np.append(m, pred)
    z = np.append(z, pred)

st.subheader('FUTURE BITCOIN PRICE')
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1, 1))
st.line_chart(z)

# About Section
st.sidebar.title('About')
st.sidebar.info(
    "This web application is for predicting cryptocurrency prices using machine learning techniques. "
    "It uses historical data to train a deep learning model which then predicts future prices."
)

# Developer Details Section
st.sidebar.title('Developer Details')
st.sidebar.info(
    "FINAL YEAR MAJOR PROJECT DONE BY: "
    "HEMANTH KUMAR S[1CG20IS019]"
    "L P SANJAY     [1CG20IS023]"
    "RAKESH P       [1CG20IS034]"
    "VIJAY A        [1CG20IS047]"
    "You can find the source code [here](https://github.com/your_username/crypto-price-prediction)."
)
