from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import src.get_dataset as get_dataset
import src.visual_cleanup as visual_cleanup
import src.model_cleanup as model_cleanup
import src.trainmodel as trainmodel
import src.streamlit as streamlit
from pathlib import Path

default_args = {
    'owner':'admin', 
    'retries': 5, 
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='ImmoEliza', 
    default_args=default_args,
    description='ImmoEliza Pipeline', 
    start_date=datetime(2023, 9, 15, 16), # Year, Month, Day, Hour
    schedule_interval='@daily'
) as dag:
    task1 = PythonOperator(
        task_id='get_data',
        python_callable=get_dataset.fetch
    )

    task2 = PythonOperator(
        task_id='clean_data_for_visual',
        python_callable=visual_cleanup.clean
    )

    task3 = PythonOperator(
        task_id='clean_data_for_model',
        python_callable=model_cleanup.clean
    )

    task4 = PythonOperator(
        task_id='train_model',
        python_callable=trainmodel.train
    )

    task5 = PythonOperator(
        task_id='streamlit',
        python_callable=streamlit.stream
    )

    task1>>task2>>task5
    task1>>task3>>task4>>task5
