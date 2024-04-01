import yaml
from models.arima import ARIMAModel
from utils.utils import get_exchange_rate_history
import joblib

# Load parameters from YAML file
with open('params.yml', 'r') as file:
    params = yaml.safe_load(file)

# Get USD to COP exchange rate history
usd_cop = get_exchange_rate_history('USD', 'COP', params['extract_start_date'], params['extract_end_date'])

# Use only the 'High' column
data = usd_cop[['High']]

# Split the data into training and testing sets
train = data[(data.index >= params['train_start_date']) & (data.index <= params['train_end_date'])]
test = data[data.index > params['train_end_date']]

# Define the ARIMA model
model = ARIMAModel(order=tuple(params['model_order']))  # Convert list to tuple

# Train the model on the USD to COP exchange rate
model.train(train)

# Create a filename that includes the model's name, order parameters, start date, and end date
model_name = f"ARIMA_{model.order}_{params['train_start_date']}_to_{params['train_end_date']}_model.pkl"

# Save the trained model to a file
joblib.dump(model, model_name)