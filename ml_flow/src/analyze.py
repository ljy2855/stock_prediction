import os
import mlflow
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from data_processing.prepare_data import prepare_backtest_input, prepare_transformer_input

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

# 모델 로드
model_name = "pytorch-transformer-time-series-model"
version = "latest"
model_uri = f"models:/{model_name}/{version}"
print(f"Loading model from URI: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully")

# MLflow Experiment 설정
mlflow.set_experiment("transformer_vector_dimension_analysis")



def analyze_vector_dimensions(feature_data, price_data, date_data, forecast_days=30):
    """Transformer 출력 벡터의 각 차원이 담고 있는 정보 분석"""
    input_tensor = prepare_transformer_input(feature_data, n_steps=30)
    prediction_vectors = []
    mean_predictions = []
    
    for i in range(forecast_days):
        input_seq = input_tensor[i].unsqueeze(0)  # (1, 30, feature_dim)
        
        # 하루 예측 수행
        prediction_vector = model.predict(input_seq.numpy())  # (1, 30)
        prediction_vector = prediction_vector.flatten()  # (30,)
        mean_prediction = np.mean(prediction_vector)  # 평균 예상 수익률
        
        prediction_vectors.append(prediction_vector)
        mean_predictions.append(mean_prediction)
    
    prediction_matrix = np.array(prediction_vectors)  # (forecast_days, 30)
    mean_predictions = np.array(mean_predictions)
    
    # 📊 1. Heatmap: 벡터 값의 시간적 변화
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        prediction_matrix.T, 
        cmap='viridis', 
        cbar=True, 
        xticklabels=date_data[:forecast_days]
    )
    plt.title("Transformer Output Vector Heatmap Over Time")
    plt.xlabel("Time (Days)")
    plt.ylabel("Vector Dimensions")
    plt.tight_layout()
    plot_path_heatmap = "report/vector_heatmap.png"
    os.makedirs("report", exist_ok=True)
    plt.savefig(plot_path_heatmap)
    plt.show()
    mlflow.log_artifact(plot_path_heatmap, artifact_path="plots")
    
    # 📊 2. Line Plot: 각 차원의 평균값 변화
    plt.figure(figsize=(14, 6))
    for i in range(prediction_matrix.shape[1]):
        plt.plot(date_data[:forecast_days], prediction_matrix[:, i], label=f"Dimension {i+1}")
    plt.title("Transformer Vector Dimension Trends Over Time")
    plt.xlabel("Time (Days)")
    plt.ylabel("Prediction Value")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plot_path_line = "report/vector_dimension_trend.png"
    plt.savefig(plot_path_line)
    plt.show()
    mlflow.log_artifact(plot_path_line, artifact_path="plots")
    
    # 📊 3. Scatter Plot: 차원과 실제 가격 변화 상관관계
    plt.figure(figsize=(14, 8))
    correlation_values = []
    for i in range(prediction_matrix.shape[1]):
        correlation = np.corrcoef(prediction_matrix[:, i], price_data[:forecast_days])[0, 1]
        correlation_values.append(correlation)
        plt.scatter(prediction_matrix[:, i], price_data[:forecast_days], label=f"Dimension {i+1}")
    
    plt.title("Transformer Vector Dimensions vs Price Changes")
    plt.xlabel("Prediction Value")
    plt.ylabel("Actual Price Change")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plot_path_scatter = "report/vector_dimension_scatter.png"
    plt.savefig(plot_path_scatter)
    plt.show()
    mlflow.log_artifact(plot_path_scatter, artifact_path="plots")
    
    # 📊 4. 각 차원의 평균 및 상관관계 기록
    dimension_summary = pd.DataFrame({
        "Dimension": [f"Dimension {i+1}" for i in range(prediction_matrix.shape[1])],
        "Mean Value": np.mean(prediction_matrix, axis=0),
        "Std Deviation": np.std(prediction_matrix, axis=0),
        "Correlation with Price": correlation_values
    })
    summary_csv = "report/vector_dimension_summary.csv"
    dimension_summary.to_csv(summary_csv, index=False)
    mlflow.log_artifact(summary_csv, artifact_path="results")
    
    print("Vector dimension analysis completed and results logged to MLflow.")


# MLflow Run 시작
with mlflow.start_run(run_name="analyze_vector_dimensions"):
    price_data, feature_data, date_data = prepare_backtest_input("data/processed/merged_data.csv", duration=60)
    
    analyze_vector_dimensions(
        feature_data=np.array(feature_data),
        price_data=np.array(price_data),
        date_data=np.array(date_data),
        forecast_days=30
    )

    print("Vector dimension analysis completed and results logged to MLflow.")
