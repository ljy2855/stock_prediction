import os
import mlflow
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data_processing.prepare_data import prepare_backtest_input

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

# 모델 로드
model_name = "pytorch-transformer-time-series-model"
version = "latest"
model_uri = f"models:/{model_name}/{version}"
print(f"Loading model from URI: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully")

# MLflow Experiment 설정
mlflow.set_experiment("transformer_signal_trend_evaluation")


def evaluate_signal_with_trend(input_sequence, feature_data, price_data, start_idx, forecast_days, date_data):
    """예상 수익률과 벡터 추세를 활용한 Buy/Sell 신호 판단 및 시각화"""
    current_sequence = input_sequence.copy()
    mean_predictions = []
    predicted_signals = []
    prediction_vectors = []
    
    # 임계값 설정
    BUY_THRESHOLD = 0.007
    SELL_THRESHOLD = -0.002

    print("Input Sequence Shape:", input_sequence.shape)
    print("First Sequence Example:\n", input_sequence[0])
    print("Mean of Sequence:", np.mean(input_sequence))
    print("Std of Sequence:", np.std(input_sequence))
    
    for day in range(forecast_days):
        input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)  # (1, 30, feature_dim)
        input_tensor = input_tensor.cpu().numpy()

        # 예측 수행
        prediction_vector = model.predict(input_tensor)
        prediction_vector = prediction_vector.flatten()
        print("Prediction Vector:", prediction_vector)
        
        prediction_vector = prediction_vector.flatten()  # (30,)
        mean_prediction = np.mean(prediction_vector)  # 평균 예상 수익률
        
        # 결과 기록
        prediction_vectors.append(prediction_vector)
        mean_predictions.append(mean_prediction)
        
        # Buy/Sell 신호 판단
        if mean_prediction > BUY_THRESHOLD:
            predicted_signals.append(1)  # Buy
        elif mean_prediction < SELL_THRESHOLD:
            predicted_signals.append(-1)  # Sell
        else:
            predicted_signals.append(0)  # Hold
        
        # 입력 시퀀스 업데이트
        next_feature = feature_data[start_idx + day]
        current_sequence = np.vstack([current_sequence[1:], next_feature])  # 업데이트된 입력 시퀀스
    
    # 📊 결과 데이터프레임 생성
    results_df = pd.DataFrame({
        "Date": date_data[start_idx:start_idx + forecast_days],
        "Actual Prices": price_data[start_idx:start_idx + forecast_days],
        "Mean Prediction": mean_predictions,
        "Predicted Signal": predicted_signals
    })
    
    # 📊 Buy/Sell 신호 시각화
    plt.figure(figsize=(14, 8))
    plt.plot(results_df["Date"], results_df["Actual Prices"], label="Actual Prices", color='blue', marker='o')
    
    for i, row in results_df.iterrows():
        if row["Predicted Signal"] == 1:
            plt.scatter(row["Date"], row["Actual Prices"], color='green', s=100, marker='^', label='Buy Signal' if i == 0 else "")
        elif row["Predicted Signal"] == -1:
            plt.scatter(row["Date"], row["Actual Prices"], color='red', s=100, marker='v', label='Sell Signal' if i == 0 else "")
    
    plt.title("Actual Prices with Buy/Sell Signals (Using Mean Prediction)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path_signal = "report/signal_prediction_graph.png"
    os.makedirs("report", exist_ok=True)
    plt.savefig(plot_path_signal)
    plt.show()
    mlflow.log_artifact(plot_path_signal, artifact_path="plots")
    


# MLflow Run 시작
with mlflow.start_run(run_name="evaluate_signal_trend"):
    price_data, feature_data, date_data = prepare_backtest_input("data/processed/merged_data.csv", duration=60)
    
    input_sequence = np.array(feature_data[:30])
    evaluate_signal_with_trend(
        input_sequence=input_sequence,
        feature_data=np.array(feature_data),
        price_data=np.array(price_data),
        start_idx=30,
        forecast_days=30,
        date_data=np.array(date_data)
    )

    print("Signal trend evaluation completed and results logged to MLflow.")
