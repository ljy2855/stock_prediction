from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
import requests
import os

# Airflow 기본 설정
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# InfluxDB 설정
INFLUXDB_URL = "http://influxdb:8086"
INFLUXDB_TOKEN = "your-influxdb-token"
INFLUXDB_ORG = "your-org"
INFLUXDB_BUCKET = "stock_bucket"

# 데이터 저장 경로
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)


# 1️⃣ 데이터 수집
def collect_data():
    stock_data = requests.get("https://api.example.com/stock").json()
    inflation_data = requests.get("https://api.example.com/inflation").json()

    # 예시 데이터프레임 생성
    stock_df = pd.DataFrame(stock_data)
    inflation_df = pd.DataFrame(inflation_data)

    # 데이터 저장
    stock_path = os.path.join(DATA_DIR, "stock_data.csv")
    inflation_path = os.path.join(DATA_DIR, "inflation_data.csv")

    stock_df.to_csv(stock_path, index=False)
    inflation_df.to_csv(inflation_path, index=False)

    print(f"✅ Data collected and saved at {stock_path}, {inflation_path}")


# 2️⃣ 데이터 전처리
def preprocess_data():
    stock_path = os.path.join(DATA_DIR, "stock_data.csv")
    inflation_path = os.path.join(DATA_DIR, "inflation_data.csv")

    stock_df = pd.read_csv(stock_path)
    inflation_df = pd.read_csv(inflation_path)

    # 예시 전처리: 결측값 처리, 스케일링
    stock_df['price'] = stock_df['price'].fillna(0).astype(float) / stock_df['price'].max()
    inflation_df['rate'] = inflation_df['rate'].fillna(0).astype(float) / inflation_df['rate'].max()

    # 전처리된 데이터 저장
    stock_df.to_csv(os.path.join(DATA_DIR, "stock_data_processed.csv"), index=False)
    inflation_df.to_csv(os.path.join(DATA_DIR, "inflation_data_processed.csv"), index=False)

    print(f"✅ Data preprocessing completed.")


# 3️⃣ 데이터 InfluxDB 삽입
def store_in_influxdb():
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=10_000))

    stock_path = os.path.join(DATA_DIR, "stock_data_processed.csv")
    inflation_path = os.path.join(DATA_DIR, "inflation_data_processed.csv")

    try:
        # 주식 데이터 삽입
        stock_df = pd.read_csv(stock_path)
        for _, row in stock_df.iterrows():
            point = Point("stock_prices") \
                .field("price", row['price']) \
                .time(row['time'])
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

        # 물가 상승률 데이터 삽입
        inflation_df = pd.read_csv(inflation_path)
        for _, row in inflation_df.iterrows():
            point = Point("inflation_rates") \
                .field("rate", row['rate']) \
                .time(row['time'])
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

        print(f"✅ Data successfully stored in InfluxDB.")

    except Exception as e:
        raise Exception(f"❌ Failed to store data in InfluxDB: {e}")
    finally:
        write_api.close()
        client.close()


# DAG 정의
with DAG(
    dag_id='data_pipeline_dag',
    default_args=default_args,
    description='Data Pipeline without FastAPI',
    schedule_interval='0 9 * * *',  # 매일 오전 9시
    start_date=datetime(2024, 6, 1),
    catchup=False
) as dag:

    task_collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=collect_data
    )

    task_preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    task_store_in_influxdb = PythonOperator(
        task_id='store_in_influxdb',
        python_callable=store_in_influxdb
    )

    task_collect_data >> task_preprocess_data >> task_store_in_influxdb
