from datetime import datetime

import mlflow
import mlflow.transformers
import mlflow.pytorch

from data_processing.prepare_data import prepare_data_for_sequences
from model.base import TimeSeriesModel
from model.adaptive_transformer import TransformerTimeSeriesModel
from utils.config import config


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("TimeSeriesModel_Experiment")
mlflow.autolog()  # PyTorch 자동 로깅 활성화

if __name__ == "__main__":

    n_steps=30 # 14일 데이터
    forecast_steps=7 # 7일 뒤의 데이터 예측
    batch_size=30

    train_loader, test_loader = prepare_data_for_sequences(n_steps=n_steps, batch_size=batch_size,forecast_steps=forecast_steps)


    # Transformer 모델 초기화
    input_size = 12  # 입력 feature 수
    d_model = 256   # 임베딩 차원
    nhead = 16       # 멀티헤드 어텐션 수
    num_layers = 8  # Transformer 레이어 수
    output_dimension = 1  # 예측 벡터 크기

    transformer_model = TransformerTimeSeriesModel(input_size, d_model, nhead, num_layers, output_dimension)
    trainer = TimeSeriesModel(transformer_model, lr=0.00000001, model_name="Adaptive_Transformer")

    # 훈련 및 평가
    with mlflow.start_run(run_name=trainer.model_name):
        trainer.train(train_loader, num_epochs=30, verbose=True, early_stopping=True, patience=10)
        trainer.evaluate(test_loader,window_size=n_steps)
