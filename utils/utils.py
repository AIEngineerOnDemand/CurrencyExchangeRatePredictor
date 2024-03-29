#from forex_python.converter import CurrencyRates
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def get_exchange_rate_history(from_currency, to_currency, start_date, end_date):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Define the file path
    file_path = os.path.join(dir_path, f'data/{from_currency}_{to_currency}_{start_date}_{end_date}.csv')

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, read it
        data = pd.read_csv(file_path)
    else:
        # If the file does not exist, download the data
        ticker_symbol = f"{from_currency}{to_currency}=X"
        data = yf.download(ticker_symbol, start=start_date, end=end_date)

        # Save the data to a CSV file
        data.to_csv(file_path)

    return data

def plot_predictions(actual_data, predicted_data, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data['Date'], actual_data['Rate'], color='black', label='Actual')
    plt.plot(predicted_data['Date'], predicted_data['Rate'], color='red', label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.title(title)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(filename)