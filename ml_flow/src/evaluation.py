import os
import mlflow
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def evaluate_signal_with_trend(input_sequence, feature_data, price_data, date_data):
    """30일 데이터를 기반으로 Buy/Sell 신호 판단 및 시각화"""
    current_sequence = input_sequence.astype(np.float32)
    mean_predictions = []
    prediction_vectors = []
    signal_dates = []
    actual_prices = []

    # 전체 데이터 순회 (start_idx: 30부터 시작)
    for start_idx in range(30, len(feature_data)):
        # 예측에 사용할 입력 시퀀스 생성
        current_sequence = feature_data[start_idx - 30:start_idx].astype(np.float32)  # (30, feature_dim)
        input_tensor = current_sequence[np.newaxis, :, :]  # (1, 30, feature_dim)

        # 예측 수행
        prediction_vector = model.predict(input_tensor)
        prediction_vector = prediction_vector.flatten()

        # 평균 예측값 계산
        mean_prediction = np.mean(prediction_vector)
        print(f"Date: {date_data[start_idx]} - Mean Prediction: {mean_prediction}")
        
        # 결과 기록
        prediction_vectors.append(prediction_vector)
        mean_predictions.append(mean_prediction)
        signal_dates.append(date_data[start_idx])
        actual_prices.append(price_data[start_idx])
        
    # 결과 시각화
    results_df = pd.DataFrame({
        "Date": pd.to_datetime(signal_dates),
        "Actual Prices": actual_prices,
        "Mean Prediction": mean_predictions,
    })
    print(results_df.tail())


    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 1) 날짜를 숫자로 변환
    x = mdates.date2num(results_df["Date"])  # 시간 -> float형
    # x축 pcolormesh에 쓸 경계(엣지) 배열 (마지막 값보다 조금 더 확장)
    x_edges = np.concatenate([x, [x[-1] + (x[-1] - x[-2])]])

    # 2) y축은 '전체 배경'을 채우기 위해, 가격의 최소/최대값 기준으로 설정
    ymin = results_df["Actual Prices"].min()
    ymax = results_df["Actual Prices"].max()
    y_edges = np.array([ymin, ymax])

    # 3) 색으로 표시할 Z(2차원). (2행, len+1 열)
    #    mean_predictions를 2행으로 복제한 뒤 마지막 열 하나 더 붙이기
    Z = np.tile(results_df["Mean Prediction"], (2, 1))  # (2, len)
    Z = np.hstack([Z, Z[:, -1].reshape(2, 1)])          # (2, len+1)

    # 4) 컬러맵 설정
    norm = plt.Normalize(min(mean_predictions), max(mean_predictions))

    # 5) pcolormesh로 배경 그라디언트
    mesh = ax1.pcolormesh(
        x_edges,             # x 축(엣지)
        y_edges,             # y 축(엣지)
        Z,                   # 색으로 표시할 값 (2D)
        cmap="Reds",         # 컬러맵
        norm=norm,           # 정규화(최솟값 ~ 최댓값)
        alpha=0.4,           # 투명도
        shading='auto'
    )

    # 6) 컬러바 추가
    cbar = plt.colorbar(mesh, ax=ax1, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label("Mean Prediction Value", fontsize=12)

    # 7) x축을 날짜 형식으로 표시
    ax1.xaxis_date()
    ax1.set_xlim(x[0], x[-1])  # x축 범위 지정(처음 ~ 끝날짜)

    # 8) 실제 가격 차트 (배경 위에 겹쳐 그림)
    ax1.plot(results_df["Date"], results_df["Actual Prices"], 
             label="Actual Prices", color='blue', marker='o', zorder=3)

    # 기타 설정
    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title("Actual Prices with Continuous Gradient Heatmap (Background)", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig("latest_model_evaluation.png")
    plt.show()
    



# MLflow Run 시작
with mlflow.start_run(run_name="evaluate_signal_trend"):
    price_data, feature_data, date_data = prepare_backtest_input("data/processed/merged_data.csv", start_date="2016-01-01", duration=3000)
    
    evaluate_signal_with_trend(
        input_sequence=feature_data,
        feature_data=np.array(feature_data),
        price_data=np.array(price_data),
        date_data=np.array(date_data)
    )

    print("Signal trend evaluation completed and results logged to MLflow.")
