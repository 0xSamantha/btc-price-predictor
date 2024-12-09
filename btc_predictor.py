import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
from pathlib import Path
import json


class BTCPredictor:
    def __init__(self):
        self.model_dir = Path('btc_model')
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / 'model.keras'
        self.history_path = self.model_dir / 'history.json'

    def quick_predict(self):
        """Quick prediction without retraining"""
        try:
            # Load existing model if available
            if not self.model_path.exists():
                print("No existing model found. Running full training...")
                return self.full_analysis()

            # Get data with proper time range for hourly data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=720)  # Stay within 730 day limit
            print(
                f"\nFetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Try hourly data first, fall back to daily if it fails
            try:
                df = yf.download('BTC-USD', start=start_date,
                               end=end_date, interval='1h')
                if df.empty:
                    raise ValueError("No hourly data available")
                print("Successfully fetched hourly data")
            except:
                print("Falling back to daily data")
                df = yf.download('BTC-USD', start=start_date, end=end_date)

            if df.empty:
                raise ValueError("Could not fetch Bitcoin data")

            print("\nData Verification:")
            print(f"Latest data timestamp: {df.index[-1]} UTC")
            print(f"Time since last update: {datetime.now() - df.index[-1]}")

            # Calculate basic indicators
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            df.dropna(inplace=True)

            print("\nRSI Analysis:")
            print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
            print(f"RSI High (24h): {df['RSI'][-24:].max():.2f}")
            print(f"RSI Low (24h): {df['RSI'][-24:].min():.2f}")

            # Prepare data
            scaler = MinMaxScaler()
            data = scaler.fit_transform(
                df[['Close', 'Volume', 'MA7', 'MA21', 'RSI']])

            print("\nPrice Movement Verification:")
            print(f"24h High: ${df['High'][-24:].max():.2f}")
            print(f"24h Low: ${df['Low'][-24:].min():.2f}")
            print(f"24h Volume: ${df['Volume'][-24:].sum():,.2f}")

            # Load model and make prediction
            model = load_model(self.model_path)
            last_60_days = data[-60:].reshape(1, 60, 5)
            prediction = model.predict(last_60_days, verbose=0)
            predicted_price = scaler.inverse_transform(
                [[prediction[0][0], 0, 0, 0, 0]])[0][0]

            # Fix: Get current price and properly convert to float
            current_price = df['Close'].iloc[-1]
            current_price_float = float(
                current_price.iloc[0]) if isinstance(current_price, pd.Series) else float(current_price)
            percent_change = ((predicted_price - current_price_float) /
                            current_price_float) * 100

            # Save prediction to history
            self.save_prediction(current_price_float,
                               predicted_price, percent_change)

            # Generate recommendation
            recommendation = self.generate_recommendation(percent_change, df)

            print("\nQuick Analysis Results:")
            print(f"Current Price: ${current_price_float:,.2f}")
            print(f"Predicted Price: ${predicted_price:,.2f}")
            print(f"Expected Change: {percent_change:.2f}%")
            print(f"Recommendation: {recommendation}")

            return predicted_price, recommendation

        except Exception as e:
            print(f"Error in quick prediction: {str(e)}")
            return None, None

    def full_analysis(self):
        """Full analysis with model retraining"""
        try:
            # Fetch data with proper time range for hourly data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=720)  # Stay within 730 day limit
            print(
                f"\nFetching training data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Try hourly data first, fall back to daily if it fails
            try:
                df = yf.download('BTC-USD', start=start_date,
                               end=end_date, interval='1h')
                if df.empty:
                    raise ValueError("No hourly data available")
                print("Successfully fetched hourly data")
            except:
                print("Falling back to daily data")
                df = yf.download('BTC-USD', start=start_date, end=end_date)

            if df.empty:
                raise ValueError("Could not fetch Bitcoin data")

            print("\nTraining Data Verification:")
            print(f"Latest data timestamp: {df.index[-1]} UTC")
            print(f"Time since last update: {datetime.now() - df.index[-1]}")

            # Calculate indicators
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            df.dropna(inplace=True)

            print("\nRSI Analysis:")
            print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
            print(f"RSI High (24h): {df['RSI'][-24:].max():.2f}")
            print(f"RSI Low (24h): {df['RSI'][-24:].min():.2f}")

            # Prepare data for training
            scaler = MinMaxScaler()
            data = scaler.fit_transform(
                df[['Close', 'Volume', 'MA7', 'MA21', 'RSI']])

            # Create sequences
            X, y = [], []
            for i in range(len(data) - 60):
                X.append(data[i:i+60])
                y.append(data[i+60, 0])

            X, y = np.array(X), np.array(y)

            # Create and train model
            model = Sequential([
                Input(shape=(60, 5)),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=20, batch_size=32, verbose=1)

            # Save model
            model.save(self.model_path)

            # Make prediction
            last_60_days = data[-60:].reshape(1, 60, 5)
            prediction = model.predict(last_60_days, verbose=0)
            predicted_price = scaler.inverse_transform(
                [[prediction[0][0], 0, 0, 0, 0]])[0][0]

            # Fix: Get current price and properly convert to float
            current_price = df['Close'].iloc[-1]
            current_price_float = float(
                current_price.iloc[0]) if isinstance(current_price, pd.Series) else float(current_price)
            percent_change = ((predicted_price - current_price_float) /
                            current_price_float) * 100

            # Save prediction to history
            try:
                self.save_prediction(current_price_float,
                                   predicted_price, percent_change)
            except Exception as e:
                print(f"Warning: Could not save prediction history: {str(e)}")

            # Generate recommendation
            recommendation = self.generate_recommendation(percent_change, df)

            print("\nFull Analysis Results:")
            print(f"Current Price: ${current_price_float:,.2f}")
            print(f"Predicted Price: ${predicted_price:,.2f}")
            print(f"Expected Change: {percent_change:.2f}%")
            print(f"Recommendation: {recommendation}")

            return predicted_price, recommendation

        except Exception as e:
            print(f"Error in full analysis: {str(e)}")
            return None, None

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_recommendation(self, percent_change, df):
        """Generate trading recommendation based on price change and RSI"""
        rsi = df['RSI'].iloc[-1]

        if rsi > 70 and percent_change < 0:
            return "SELL - Overbought conditions and downward trend expected"
        elif rsi < 30 and percent_change > 0:
            return "BUY - Oversold conditions and upward trend expected"
        elif abs(percent_change) < 2:
            return "HOLD - No significant movement expected"
        elif percent_change > 2:
            return f"BUY - Upward trend of {percent_change:.1f}% expected"
        else:
            return f"SELL - Downward trend of {abs(percent_change):.1f}% expected"

    def save_prediction(self, current_price, predicted_price, percent_change):
        """Save prediction to history file"""
        try:
            # Initialize history with empty list if file doesn't exist
            if not self.history_path.exists():
                history = []
            else:
                try:
                    with open(self.history_path, 'r') as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    # If JSON is invalid, start fresh
                    history = []

            # Ensure history is a list
            if not isinstance(history, list):
                history = []

            # Create new prediction entry
            new_prediction = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'percent_change': float(percent_change)
            }

            # Append new prediction
            history.append(new_prediction)

            # Keep only last 30 predictions
            if len(history) > 30:
                history = history[-30:]

            # Write to file with error handling
            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=4)

        except Exception as e:
            print(f"Error saving prediction history: {str(e)}")
            # Don't recreate the file here, just log the error


def main():
    predictor = BTCPredictor()

    while True:
        try:
            print("\nBitcoin Price Predictor")
            print("1. Quick Prediction")
            print("2. Full Analysis")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == '1':
                predictor.quick_predict()
            elif choice == '2':
                predictor.full_analysis()
            elif choice == '3':
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting...")
            break
        except EOFError:
            print("\nInput error. Please run from a regular terminal.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()