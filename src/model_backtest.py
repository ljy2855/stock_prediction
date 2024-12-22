import torch
from model.transformer import TransformerTimeSeriesModel
import pandas as pd

# Step 1: Initialize and load Transformer model
input_size = 8
d_model = 128
nhead = 4
num_layers = 6
output_size = 30
model_path = "models/Transformer_2d3da52c42714d09966f4533c87c490c.pth"

model = TransformerTimeSeriesModel(input_size, d_model, nhead, num_layers, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

print("Model loaded successfully")

# Step 2: Load historical test data for backtesting
from data_processing.prepare_data import prepare_backtest_input

import numpy as np


price_data, feature_data, date_data = prepare_backtest_input()

# Step 3: Run backtest
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
)

agent.run()

# Step 4: Visualize backtest results
agent.save_report()
