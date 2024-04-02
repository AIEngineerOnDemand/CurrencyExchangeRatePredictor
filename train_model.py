from models.arima import ARIMAModel
from utils.utils import get_exchange_rate_history, split_data
import yaml

def train_model(model_order, train_start_date, train_end_date, extract_end_date):
    # Convert model_order from string to list of integers
    model_order = [int(x) for x in yaml.safe_load(model_order)]

    # Get USD to COP exchange rate history
    usd_cop = get_exchange_rate_history('USD', 'COP', train_start_date, extract_end_date)

    # Use only the 'High' column
    data = usd_cop[['High']]

    # Split the data into training and testing sets
    train, test = split_data(data, train_start_date, train_end_date)

    # Define the ARIMA model
    model = ARIMAModel(order=tuple(model_order))  # Convert list to tuple

    # Train the model on the USD to COP exchange rate
    model.train(train)

    # Create a filename that includes the model's name, order parameters, start date, and end date
    model_name = f"ARIMA_{model.order}_{train_start_date}_to_{train_end_date}_model.pkl"

    # Save the trained model to a file
    model.save(model_name)
    
    # Return the model name
    return model_name