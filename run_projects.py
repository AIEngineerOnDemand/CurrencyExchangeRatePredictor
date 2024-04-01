import yaml
import mlflow

# Load parameters from YAML file
with open('params.yml', 'r') as file:
    params = yaml.safe_load(file)

# Run the training script as an MLflow project
print("Starting training...")
result = mlflow.run("train_model.py", parameters=params)
print(f"Training completed with status: {result.get_status()}")

# Define the parameters for the inference script
infer_params = {
    'model_name': f"ARIMA_{tuple(params['model_order'])}_{params['train_start_date']}_to_{params['train_end_date']}_model.pkl",
    'start_date': params['extract_end_date'],
    'end_date': params['extract_end_date']
}

# Run the inference script as an MLflow project
print("Starting inference...")
result = mlflow.run("main.py", parameters=infer_params)
print(f"Inference completed with status: {result.get_status()}")