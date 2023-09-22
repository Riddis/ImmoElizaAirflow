from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import src.get_dataset as get_dataset
import src.visual_cleanup as visual_cleanup
import src.model_cleanup as model_cleanup
import src.trainmodel as trainmodel
import src.webapp as app
from pathlib import Path

default_args = {
    'owner':'admin', 
    'retries': 5, 
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='ImmoEliza_Debug', 
    default_args=default_args,
    description='ImmoEliza Pipeline Debug', 
    start_date=datetime(2023, 9, 15, 16), # Year, Month, Day, Hour
    schedule_interval='@daily'
) as dag:

    task5 = BashOperator(
        task_id='streamlit',
        bash_command='streamlit run /mnt/c/users/ridd/documents/repos/immoelizaairflow/airflow/dags/src/webapp.py'
        #bash_command = 'streamlit run /home/ridd/airflow_env/bin/airflow'
    )

    task5

