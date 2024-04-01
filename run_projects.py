import yaml
import mlflow

# Load parameters from YAML file
with open('params.yml', 'r') as file:
    params = yaml.safe_load(file)

# Run the training script as an MLflow project
mlflow.run("train_model.py", parameters=params)

# Define the parameters for the inference script
infer_params = {
    'model_name': f"ARIMA_{tuple(params['model_order'])}_{params['train_start_date']}_to_{params['train_end_date']}_model.pkl",
    'start_date': params['extract_end_date'],
    'end_date': params['extract_end_date']
}  # Added closing brace here

# Run the inference script as an MLflow project
mlflow.run("main.py", parameters=infer_params)