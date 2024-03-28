from forex_python.converter import CurrencyRates
import pandas as pd

def get_exchange_rate_history(base_currency, target_currency, start_date, end_date):
    currency_rates = CurrencyRates()

    date_range = pd.date_range(start=start_date, end=end_date)
    exchange_rates = []

    for date in date_range:
        try:
            rate = currency_rates.get_rate(base_currency, target_currency, date)
            exchange_rates.append((date, rate))
        except:
            print(f"Could not get rate for {date}")

    return pd.DataFrame(exchange_rates, columns=['Date', 'Rate'])

