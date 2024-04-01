from models.arima import ARIMAModel
from utils.utils import get_exchange_rate_history
import joblib

# Get USD to COP exchange rate history
start_date, end_date = '2023-01-01', '2024-03-28'
usd_cop = get_exchange_rate_history('USD', 'COP', start_date, end_date)

# Use only the 'High' column
data = usd_cop[['High']]

# Split the data into training and testing sets
train = data[data.index.year < 2024]

# Define the ARIMA model
model = ARIMAModel(order=(1, 1, 1))  # Adjust the order parameters as needed

# Train the model on the USD to COP exchange rate
model.train(train)

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')