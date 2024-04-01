from utils.utils import plot_predictions, parse_common_args, split_data
from models.arima import ARIMAModel
from utils.utils import get_exchange_rate_history
import pandas as pd

def main():
    # Parse command line arguments
    args = parse_common_args()

    # Define the ARIMA model
    model = ARIMAModel(order=args.model_order)

    # Load the ARIMA model from the file
    model.load(args.model_name)

    # Get USD to COP exchange rates for the specified dates
    usd_cop = get_exchange_rate_history('USD', 'COP', args.extract_start_date, args.extract_end_date)

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
    train_last_month = train[(train.index.year == int(args.train_end_date[:4])) & (train.index.month == int(args.train_end_date[5:7]))]

    # Plot the actual and predicted exchange rates
    # Plot the actual and predicted exchange rates
    plot_predictions(train_last_month, test, predicted_data, 'USD to COP Exchange Rate', 'results/usd_cop_predictions.png')

if __name__ == "__main__":
    main()