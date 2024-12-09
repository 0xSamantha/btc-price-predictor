import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def calculate_technical_indicators(df):
    """Calculate RSI and Moving Averages"""
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate RSI
    window = 14
    # Get gains and losses
    gains = df['Returns'].copy()
    losses = df['Returns'].copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    
    return df

def fetch_bitcoin_data():
    """fetch Bitcoin historical data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    btc = calculate_technical_indicators(btc)
    return btc

def prepare_data(df, lookback=60, split=0.8):
    """prepare data for LSTM model with price, volume, and technical indicators"""
    # Get all features
    price_data = df['Close'].values.reshape(-1, 1)
    volume_data = df['Volume'].values.reshape(-1, 1)
    rsi_data = df['RSI'].values.reshape(-1, 1)
    ma7_data = df['MA7'].values.reshape(-1, 1)
    ma21_data = df['MA21'].values.reshape(-1, 1)
    
    # Normalize features
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    rsi_scaler = MinMaxScaler()
    ma_scaler = MinMaxScaler()
    
    price_normalized = price_scaler.fit_transform(price_data)
    volume_normalized = volume_scaler.fit_transform(volume_data)
    rsi_normalized = rsi_scaler.fit_transform(rsi_data)
    ma7_normalized = ma_scaler.fit_transform(ma7_data)
    ma21_normalized = ma_scaler.fit_transform(ma21_data)
    
    # Combine all features
    combined_data = np.hstack((
        price_normalized,
        volume_normalized,
        rsi_normalized,
        ma7_normalized,
        ma21_normalized
    ))
    
    # Create sequences
    X, y = [], []
    for i in range(len(combined_data) - lookback):
        X.append(combined_data[i:i+lookback])
        y.append(price_normalized[i+lookback])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, price_scaler, combined_data

def create_model(lookback):
    """create LSTM model with additional features"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 5)),  # Changed to 5 features
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate():
    """main function to train and evaluate the model"""
    # Fetch data
    print("Fetching Bitcoin data...")
    df = fetch_bitcoin_data()
    
    # prepare data
    print("Preparing data...")
    lookback = 60
    X_train, X_test, y_train, y_test, price_scaler, combined_data = prepare_data(df, lookback)
    
    # create and train model
    print("Training model...")
    model = create_model(lookback)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # make predictions
    print("\nMaking predictions...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # inverse transform predictions
    train_predictions = price_scaler.inverse_transform(train_predictions)
    y_train_inv = price_scaler.inverse_transform(y_train)
    test_predictions = price_scaler.inverse_transform(test_predictions)
    y_test_inv = price_scaler.inverse_transform(y_test)
    
    # Calculate errors
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
    train_mae = mean_absolute_error(y_train_inv, train_predictions)
    test_mae = mean_absolute_error(y_test_inv, test_predictions)
    
    print("\nResults:")
    print(f"Train RMSE: ${train_rmse:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Train MAE: ${train_mae:.2f}")
    print(f"Test MAE: ${test_mae:.2f}")

    # Make next day prediction
    print("\nMaking next day prediction...")
    last_sequence = combined_data[-lookback:].reshape(1, lookback, 5)
    next_day = model.predict(last_sequence)
    next_day_price = price_scaler.inverse_transform(next_day)[0][0]
    
    print(f"\nPredicted next day Bitcoin price: ${next_day_price:,.2f}")
    
    # Show recent predictions vs actual prices
    print("\nLast 5 days comparison:")
    for i in range(-5, 0):
        print(f"Actual: ${y_test_inv[i][0]:,.2f}")
        print(f"Predicted: ${test_predictions[i][0]:,.2f}")
        print(f"Difference: ${abs(y_test_inv[i][0] - test_predictions[i][0]):,.2f}")
        print()
    
    return model, history, test_predictions, y_test_inv

if __name__ == "__main__":
    model, history, predictions, actual = train_and_evaluate()