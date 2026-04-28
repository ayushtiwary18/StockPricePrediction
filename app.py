import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

# PAGE CONFIG

st.set_page_config(page_title="Stock AI Dashboard", layout="wide")


# LOAD MODEL + SCALER

model = load_model('Stock Predictions Model.keras')
scaler = pickle.load(open('scaler.pkl', 'rb'))


# HEADER

st.title("Stock Price Prediction Dashboard")
st.markdown("LSTM-based time series forecasting system")


# SIDEBAR CONTROLS

st.sidebar.header("Controls")

popular_stocks = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA", "META", "NFLX"]

stock_mode = st.sidebar.radio("Stock Input Mode", ["Popular Stocks", "Custom Stock"])

if stock_mode == "Popular Stocks":
    stock = st.sidebar.selectbox("Select Stock", popular_stocks)
else:
    stock = st.sidebar.text_input("Enter Stock Symbol (e.g. INFY.NS, TCS.NS)", "AAPL")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

run = st.sidebar.button("🚀 Run Prediction")


if run:

    # LOAD DATA
  
    data = yf.download(stock, start, end)

    if data.empty:
        st.error("No data found. Check stock symbol or date range.")
        st.stop()

    close_series = data['Close']

    current_price = float(close_series.iloc[-1])
    prev_price = float(close_series.iloc[-2])
    change = current_price - prev_price

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"${current_price:.2f}", f"{change:.2f}")
    col2.metric("Total Days", len(data))
    col3.metric("Model", "LSTM")


    # MOVING AVERAGES

    ma50 = close_series.rolling(50).mean()
    ma100 = close_series.rolling(100).mean()

    st.subheader("📈 Stock Trend Analysis")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(close_series, label="Close Price")
    ax.plot(ma50, label="MA50")
    ax.plot(ma100, label="MA100")
    ax.legend()
    st.pyplot(fig)


    # PREPROCESSING

    data_train = pd.DataFrame(close_series[:int(len(close_series)*0.8)])
    data_test = pd.DataFrame(close_series[int(len(close_series)*0.8):])

    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)

    data_test_scale = scaler.fit_transform(data_test)

    x = []
    y = []

    for i in range(100, len(data_test_scale)):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    # PREDICTION

    predict = model.predict(x, verbose=0)
    predict = scaler.inverse_transform(predict)
    y = scaler.inverse_transform(y.reshape(-1, 1))

    # ACTUAL VS PREDICTED

    st.subheader("📊 Actual vs Predicted Price")

    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(y, label="Actual Price", color="green")
    ax2.plot(predict, label="Predicted Price", color="red")
    ax2.legend()
    st.pyplot(fig2)

    # SIGNAL

    trend = predict[-1] - predict[-2]

    st.subheader("📢 Trading Signal")

    if trend > 0:
        st.success("BUY 📈 (Upward trend expected)")
    else:
        st.error("SELL 📉 (Downward trend expected)")

    # 7-DAY FUTURE PREDICTION

    st.subheader("🔮 7-Day Future Prediction")

    future_input = data_test_scale[-100:]
    temp_input = future_input.reshape(-1).tolist()

    future_output = []

    for i in range(7):
        x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)

        pred = model.predict(x_input, verbose=0)
        value = pred[0][0]

        future_output.append(value)
        temp_input.append(value)

    future_output = np.array(future_output).reshape(-1, 1)
    future_output = scaler.inverse_transform(future_output)

    future_dates = pd.date_range(start=data.index[-1], periods=8, freq='D')[1:]

    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(future_dates, future_output, marker='o', label="Future Prediction")
    ax3.set_title("Next 7 Days Forecast")
    ax3.legend()
    st.pyplot(fig3)

