import argparse
import yaml
from models.arima import ARIMAModel
from utils.utils import get_exchange_rate_history
import joblib

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--extract_start_date', type=str)
parser.add_argument('--extract_end_date', type=str)
parser.add_argument('--train_start_date', type=str)
parser.add_argument('--train_end_date', type=str)
parser.add_argument('--model_order', type=str)  # Pass as a string, e.g., "[1, 1, 1]"
args = parser.parse_args()

# Convert model_order from string to list of integers
model_order = [int(x) for x in yaml.safe_load(args.model_order)]

# Get USD to COP exchange rate history
usd_cop = get_exchange_rate_history('USD', 'COP', args.extract_start_date, args.extract_end_date)

# Use only the 'High' column
data = usd_cop[['High']]

# Split the data into training and testing sets
train = data[(data.index >= args.train_start_date) & (data.index <= args.train_end_date)]
test = data[data.index > args.train_end_date]

# Define the ARIMA model
model = ARIMAModel(order=tuple(model_order))  # Convert list to tuple

# Train the model on the USD to COP exchange rate
model.train(train)

# Create a filename that includes the model's name, order parameters, start date, and end date
model_name = f"ARIMA_{model.order}_{args.train_start_date}_to_{args.train_end_date}_model.pkl"

# Save the trained model to a file
joblib.dump(model, model_name)