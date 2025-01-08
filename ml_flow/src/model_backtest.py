from matplotlib import pyplot as plt
import numpy as np
import torch
from model.transformer import TransformerTimeSeriesModel

import pandas as pd


# Step 2: Initialize and load Transformer model
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

model_name = "pytorch-transformer-time-series-model"
version = 9

model_uri = f"models:/{model_name}/{version}"
print(f"ℹ️ Loading model from URI: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully")

# Step 3: Perform predictions on historical data
from data_processing.prepare_data import prepare_backtest_input


mlflow.set_experiment("transformer_backtest_evaluation")

# Step 5: Run backtest
from backtest.agent import BacktestAgent


# MLflow 실험 시작
mlflow.set_experiment("transformer_backtest_evaluation")

# MLflow Run 시작
with mlflow.start_run(run_name="backtest_evaluation"):

    total_returns = []
    evaluation_features = []
    evaluation_targets = []

    num_runs = 30  # 백테스트 반복 횟수

    for i in range(num_runs):
        # 데이터 준비
        price_data, feature_data, date_data = prepare_backtest_input("data/processed/merged_data.csv", duration=180)

        initial_balance = price_data[0]

        agent = BacktestAgent(
            model=model,
            price_data=price_data,
            feature_data=feature_data,
            date_data=date_data,
            n_steps=30,
            initial_balance=initial_balance,
            transaction_cost=0.001,
            signal_threshold=0.01,
            risk_tolerance=0.3,
        )

        agent.run()
        total_returns.append(agent.total_return)

        print(f"Run {i+1}/{num_runs}: Total Return = {agent.total_return:.4f}")

    # 메트릭 로깅
    avg_return = np.mean(total_returns)
    max_return = np.max(total_returns)
    min_return = np.min(total_returns)
    std_return = np.std(total_returns)

    mlflow.log_metric("average_return", avg_return)
    mlflow.log_metric("max_return", max_return)
    mlflow.log_metric("min_return", min_return)
    mlflow.log_metric("std_return", std_return)

    # 그래프 저장
    plt.figure(figsize=(12, 6))
    plt.hist(total_returns, bins=10, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Returns Across Multiple Backtests')
    plt.xlabel('Total Return')
    plt.ylabel('Frequency')
    plt.savefig("return_distribution.png")
    mlflow.log_artifact("return_distribution.png")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_runs + 1), total_returns, marker='o', linestyle='-')
    plt.title('Total Return Across Multiple Backtests')
    plt.xlabel('Backtest Run')
    plt.ylabel('Total Return')
    plt.grid(True)
    plt.savefig("return_trend.png")
    mlflow.log_artifact("return_trend.png")