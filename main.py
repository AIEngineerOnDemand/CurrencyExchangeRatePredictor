from utils.utils import plot_predictions, parse_common_args, split_data
from models.arima import ARIMAModel
from utils.utils import get_exchange_rate_history
import pandas as pd

def run_inference(model_name, train_start_date, train_end_date, extract_start_date, extract_end_date):
    # Define the ARIMA model
    model = ARIMAModel(order=None)  # Order is not needed for inference

    # Load the ARIMA model from the file
    model.load(model_name)

    # Get USD to COP exchange rates for the specified dates
    usd_cop = get_exchange_rate_history('USD', 'COP', extract_start_date, extract_end_date)

    # Use only the 'High' column
    data = usd_cop[['High']]

    # Split the data into training and testing sets
    train, test = split_data(data, train_start_date, train_end_date)

    # Make predictions for the test data
    predictions = model.predict(test.index[0], test.index[-1])

    # Create a DataFrame for the predictions
    predicted_data = pd.DataFrame({
        'Date': test.index,
        'Rate': predictions
    })

    # Filter the training data to only include the last month of the training period
    train_last_month = train[(train.index.year == int(train_end_date[:4])) & (train.index.month == int(train_end_date[5:7]))]

    # Plot the actual and predicted exchange rates
    plot_predictions(train_last_month, test, predicted_data, 'USD to COP Exchange Rate', 'results/usd_cop_predictions.png')