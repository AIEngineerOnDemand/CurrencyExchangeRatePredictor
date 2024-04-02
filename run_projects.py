from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from train_model import train_model
from main import run_inference
import yaml

# Load parameters from params.yml
with open('params.yml', 'r') as f:
    params = yaml.safe_load(f)

# Define the DAG
dag = DAG('currency_exchange_rate_predictor', description='A DAG that trains and deploys a currency exchange rate prediction model',
          schedule_interval='0 0 * * *',
          start_date=datetime(2022, 1, 1), catchup=False)

# Define the training task
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs=params,
    dag=dag,
    do_xcom_push=True  # This makes the task return its result (the model name)
)


# Define the inference task
run_inference_task = PythonOperator(
    task_id='run_inference',
    python_callable=run_inference,
    op_kwargs=params,
    dag=dag
)

# Set the task dependencies
train_model_task >> run_inference_task

# Get the model name from the train_model_task
model_name = train_model_task.output

# Pass the model name to the run_inference_task
run_inference_task.op_kwargs['model_name'] = model_name