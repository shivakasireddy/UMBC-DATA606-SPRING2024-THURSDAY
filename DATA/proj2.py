import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Streamlit app title
st.title('Comprehensive Stock Data Analysis and Prediction')

# Dropdown menu for user to select what to do
option = st.selectbox(
    'Select the operation you want to perform',
    ('Visualize Stock Data', 'Predict Microsoft Stock Price Based on S&P 500', 'Linear Regression of JPM on S&P 500', 'Visualize Moving Averages', 'Random Forest Stock Prediction')
)

# Function to download stock data
def get_data(ticker, period=None, start=None, end=None):
    if period:
        return yf.download(ticker, period=period)['Close']
    else:
        return yf.download(ticker, start=start, end=end)['Adj Close']

# Function to create features for machine learning
def create_features(data):
    data['Prev Close'] = data['Close'].shift(1)
    data['1-day change'] = data['Close'].diff()
    data['5-day average'] = data['Close'].rolling(window=5).mean()
    data['5-day volatility'] = data['Close'].rolling(window=5).std()
    data.dropna(inplace=True)
    return data

# Function to prepare the dataset for training
def prepare_data(data):
    X = data[['Prev Close', '1-day change', '5-day average', '5-day volatility']]
    y = data['Close']
    return train_test_split(X, y, test_size=0.2, shuffle=False)

if option == 'Visualize Stock Data':
    # Dropdown menu for user to select stock for visualization
    stock_option = st.selectbox(
        'Which stock data do you want to visualize?',
        ('S&P 500', 'Microsoft', 'JPMorgan')
    )

    # Mapping from options to Yahoo Finance tickers
    tickers = {
        'S&P 500': '^GSPC',
        'Microsoft': 'MSFT',
        'JPMorgan': 'JPM'
    }

    # Load and display stock data
    data = get_data(tickers[stock_option], period='10y')
    st.write(f"Displaying {stock_option} data")
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(f"{stock_option} Closing Price Over Time")
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

elif option == 'Predict Microsoft Stock Price Based on S&P 500':
    with st.spinner('Downloading stock data...'):
        sp500_close = get_data('^GSPC', period='10y').rename('SP500_Close')
        msft_close = get_data('MSFT', period='10y').rename('MSFT_Close')
        data = pd.concat([sp500_close, msft_close], axis=1)
        data.dropna(inplace=True)

    # Display data head
    st.write("Displaying first few rows of the dataset:")
    st.dataframe(data.head())

    # Regression analysis
    X = sm.add_constant(data['SP500_Close'])
    Y = data['MSFT_Close']
    model = sm.OLS(Y, X).fit()

    # Showing the summary of the model
    st.write("Regression Model Summary:")
    st.text(str(model.summary()))

    # Predicting Microsoft close prices
    data['Predicted_MSFT_Close'] = model.predict(X)

    # Plotting
    st.write("Plotting Actual vs Predicted Microsoft Prices:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['SP500_Close'], data['MSFT_Close'], color='blue', label='Actual MSFT Prices')
    ax.plot(data['SP500_Close'], data['Predicted_MSFT_Close'], color='red', linewidth=2, label='Predicted MSFT Prices')
    ax.set_title('Actual vs Predicted Microsoft Prices')
    ax.set_xlabel('S&P 500 Close Price')
    ax.set_ylabel('Microsoft Close Price')
    ax.legend()
    st.pyplot(fig)

elif option == 'Linear Regression of JPM on S&P 500':
    # Downloading data
    with st.spinner('Downloading stock data...'):
        jpm_data = get_data('JPM', start='2014-01-01', end='2024-01-01')
        sp500_data = get_data('^GSPC', start='2014-01-01', end='2024-01-01')

    # Ensure the data aligns
    data = pd.DataFrame({
        'JPM': jpm_data,
        'SP500': sp500_data
    })
    data.dropna(inplace=True)

    # Prepare data for regression
    X = data['SP500'].values.reshape(-1, 1)  # Features
    y = data['JPM'].values.reshape(-1, 1)    # Target variable

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Actual Price')
    ax.plot(X, y_pred, color='red', label='Fitted Line')
    plt.title('Linear Regression of JPM on S&P 500')
    plt.xlabel('S&P 500')
    plt.ylabel('JPM')
    plt.legend()
    st.pyplot(fig)

    # Print model statistics
    st.write(f"Coefficient: {model.coef_[0][0]}")
    st.write(f"Intercept: {model.intercept_[0]}")
    st.write(f"Mean squared error: {mean_squared_error(y, y_pred)}")
    st.write(f"R^2 score: {r2_score(y, y_pred)}")

elif option == 'Visualize Moving Averages':
    # Downloading data for moving averages
    with st.spinner('Downloading stock data for moving averages...'):
        sp500 = get_data('^GSPC', period='10y').rename('SP500_Close')
        msft = get_data('MSFT', period='10y').rename('MSFT_Close')

    # Calculating moving averages
    sp500_sma50 = sp500.rolling(window=50).mean()
    sp500_sma200 = sp500.rolling(window=200).mean()
    msft_sma50 = msft.rolling(window=50).mean()
    msft_sma200 = msft.rolling(window=200).mean()

    sp500_ema50 = sp500.ewm(span=50, adjust=False).mean()
    sp500_ema200 = sp500.ewm(span=200, adjust=False).mean()
    msft_ema50 = msft.ewm(span=50, adjust=False).mean()
    msft_ema200 = msft.ewm(span=200, adjust=False).mean()

    # Plotting S&P 500 moving averages
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(sp500, label='S&P 500 Close', color='black')
    ax.plot(sp500_sma50, label='50-Day SMA', color='green')
    ax.plot(sp500_sma200, label='200-Day SMA', color='red')
    ax.plot(sp500_ema50, label='50-Day EMA', color='blue')
    ax.plot(sp500_ema200, label='200-Day EMA', color='orange')
    plt.title('S&P 500 Close Prices and Moving Averages')
    plt.legend()
    st.pyplot(fig)

    # Plotting Microsoft moving averages
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(msft, label='Microsoft Close', color='black')
    ax.plot(msft_sma50, label='50-Day SMA', color='green')
    ax.plot(msft_sma200, label='200-Day SMA', color='red')
    ax.plot(msft_ema50, label='50-Day EMA', color='blue')
    ax.plot(msft_ema200, label='200-Day EMA', color='orange')
    plt.title('Microsoft Close Prices and Moving Averages')
    plt.legend()
    st.pyplot(fig)

elif option == 'Random Forest Stock Prediction':
    # Fetch historical data
    with st.spinner('Downloading stock data for prediction...'):
        sp500_data = yf.download('^GSPC', period='2y')
        msft_data = yf.download('MSFT', period='2y')

    # Create features
    sp500_features = create_features(sp500_data)
    msft_features = create_features(msft_data)

    # Prepare the dataset for training
    X_train_sp500, X_test_sp500, y_train_sp500, y_test_sp500 = prepare_data(sp500_features)
    X_train_msft, X_test_msft, y_train_msft, y_test_msft = prepare_data(msft_features)

    # Initialize and train the Random Forest model
    rf_model_sp500 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_sp500.fit(X_train_sp500, y_train_sp500)

    rf_model_msft = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_msft.fit(X_train_msft, y_train_msft)

    # Predictions
    predictions_sp500 = rf_model_sp500.predict(X_test_sp500)
    predictions_msft = rf_model_msft.predict(X_test_msft)

    # Evaluate the model
    mse_sp500 = mean_squared_error(y_test_sp500, predictions_sp500)
    mse_msft = mean_squared_error(y_test_msft, predictions_msft)
    st.write(f'Mean Squared Error for S&P 500: {mse_sp500}')
    st.write(f'Mean Squared Error for Microsoft: {mse_msft}')

    # Plotting the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_sp500.index, y_test_sp500, label='Actual S&P 500', color='blue')
    ax.plot(y_test_sp500.index, predictions_sp500, label='Predicted S&P 500', color='red')
    ax.set_title('S&P 500 Actual vs Predicted')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_msft.index, y_test_msft, label='Actual Microsoft', color='blue')
    ax.plot(y_test_msft.index, predictions_msft, label='Predicted Microsoft', color='red')
    ax.set_title('Microsoft Actual vs Predicted')
    ax.legend()
    st.pyplot(fig)
