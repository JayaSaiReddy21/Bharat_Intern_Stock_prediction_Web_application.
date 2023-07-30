import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import yfinance as yf



start = '2013-01-01'
end = '2022-12-31'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'GOOG')
df = yf.download(user_input, start=start, end=end)

#Describing the data
st.subheader('Data from 2013 to 2022')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df['Close'], 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

#Splitting and scaling the data

data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.70):])
scaler = MinMaxScaler()
data_train_new = scaler.fit_transform(data_train)



#Loading the model

model = load_model('C:/Users/cgkck/Downloads/Bharat Intern/keras_model.h5')


#Testing part

past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scale_factor = 1/0.0102168
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

scale_factor = 1/scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predicted Price')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(range(700), y_test[:700], 'b', label='Original')
# Plot predicted stock prices starting from index 700
plt.plot(range(700, len(y_pred)), y_pred[700:], 'r', label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual and Predicted Stock Prices')
plt.legend()
st.pyplot(fig2)