import torch
from model.transformer import TransformerTimeSeriesModel

import pandas as pd

# Step 2: Initialize and load Transformer model
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

model_name = "pytorch-transformer-time-series-model"
version = 8

model_uri = f"models:/{model_name}/{version}"
print(f"ℹ️ Loading model from URI: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully")

# Step 3: Perform predictions on historical data
from data_processing.prepare_data import prepare_backtest_input

price_data, feature_data, date_data = prepare_backtest_input("data/processed/merged_data.csv")

# Step 5: Run backtest
from backtest.agent import BacktestAgent

initial_balance = price_data[0]

# 백테스트 실행
n_steps = 30  # 시퀀스 길이
agent = BacktestAgent(
    model=model,
    price_data=price_data,
    feature_data=feature_data,
    date_data=date_data,
    n_steps=n_steps,
    initial_balance=initial_balance,
    transaction_cost=0.001,
    signal_threshold=0.01,
    risk_tolerance=0.3,
)

agent.run()

agent.show_plot()