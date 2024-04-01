import yaml
from datetime import datetime, timedelta
from utils.utils import get_exchange_rate_history, split_data, parse_common_args
from models.arima import ARIMAModel
from utils.utils import plot_predictions 
import pandas as pd

def main():
    
    args = parse_common_args()
    
    # Load parameters from YAML file
    with open('params.yml', 'r') as file:
        params = yaml.safe_load(file)

    # Define the ARIMA model
    model = ARIMAModel(order=tuple(params['model_order']))

    # Load the ARIMA model from the file
    model_name = f"ARIMA_{model.order}_{params['train_start_date']}_to_{params['train_end_date']}_model.pkl"
    model.load(model_name)

    # Get USD to COP exchange rates for the specified dates
    usd_cop = get_exchange_rate_history('USD', 'COP', params['extract_start_date'], params['extract_end_date'])

    # Use only the 'High' column
    data = usd_cop[['High']]

    # Split the data into training and testing sets
    train, test = split_data(data, args.train_start_date, args.train_end_date)

    # Make predictions for the test data
    predictions = model.predict(test.index[0], test.index[-1])

    # Create a DataFrame for the predictions
    predicted_data = pd.DataFrame({
        'Date': test.index,
        'Rate': predictions
    })

    # Filter the training data to only include the last month of the training period
    train_last_month = train[(train.index.year == int(params['train_end_date'][:4])) & (train.index.month == int(params['train_end_date'][5:7]))]

    # Plot the actual and predicted exchange rates
    plot_predictions(train_last_month, test, predicted_data, 'USD to COP Exchange Rate', 'usd_cop_predictions.png')

if __name__ == "__main__":
    main()