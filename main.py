from datetime import datetime, timedelta
from utils.utils import get_exchange_rate_history
from models.arima import ARIMAModel
from utils.utils import plot_predictions 
import pandas as pd

# Get USD to COP exchange rates for the past 2 years
start_date, end_date = '2023-01-01', '2024-03-28'
usd_cop = get_exchange_rate_history('USD', 'COP', start_date, end_date)

# Use only the 'High' column
data = usd_cop[['High']]

# Split the data into training and testing sets
train = data[data.index.year < 2024]
test = data[(data.index.year == 2024) & (data.index.month == 1) & (data.index.day <= 7)]

# Define the ARIMA model
model = ARIMAModel(order=(1, 1, 1))  # Adjust the order parameters as needed

# Train the model on the USD to COP exchange rate
model.train(train)

# Make predictions for the test data
predictions = model.predict(test.index[0], test.index[-1])

# Create a DataFrame for the predictions
predicted_data = pd.DataFrame({
    'Date': test.index,
    'Rate': predictions
})

# Plot the actual and predicted exchange rates
plot_predictions(train, test, predicted_data, 'USD to COP Exchange Rate', 'usd_cop_predictions.png')