from datetime import datetime

import mlflow
import mlflow.transformers
from mlflow.models import infer_signature
import mlflow.pytorch

from data_processing.prepare_data import prepare_data_for_sequences
from data_processing.stock_data import download_stock_data
from data_processing.fred_data import download_economic_data
from data_processing.transform_data import merge_data
from model.base import TimeSeriesModel
from model.transformer import TransformerTimeSeriesModel
from utils.config import config


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("TimeSeriesModel_Experiment")
mlflow.autolog()  # PyTorch 자동 로깅 활성화

START_DATE = "2015-01-01"
TICKER = "SPY"
END_DATE = datetime.today().strftime('%Y-%m-%d')


def init_data():
    # 경로 설정
    stock_data_path = config.get_path("paths", "raw_data", "stock_data")
    
    # 데이터 다운로드 및 저장
    stock_data = download_stock_data(TICKER, START_DATE, END_DATE, stock_data_path)

    # 경제 데이터 다운로드 및 저장
    rate_data, cpi_data = download_economic_data(START_DATE, END_DATE, config.get_path("paths", "raw_data", "economic_data"))

    # 데이터 병합
    merged_data = merge_data(stock_data, rate_data, cpi_data, config.get_path("paths", "processed_data", "stock_data"))

if __name__ == "__main__":
    # init_data()


    train_loader, test_loader = prepare_data_for_sequences(n_steps=30, batch_size=64)

    # Transformer 모델 초기화
    input_size = 8  # 입력 feature 수
    d_model = 128   # 임베딩 차원
    nhead = 4       # 멀티헤드 어텐션 수
    num_layers = 8  # Transformer 레이어 수
    forecast_steps = 30  # 예측 벡터 크기

    transformer_model = TransformerTimeSeriesModel(input_size, d_model, nhead, num_layers, forecast_steps)
    trainer = TimeSeriesModel(transformer_model, lr=0.001, model_name="Transformer")

    # 훈련 및 평가
    with mlflow.start_run(run_name=trainer.model_name):
        trainer.train(train_loader, num_epochs=10, verbose=True, early_stopping=True, patience=10)
        trainer.evaluate(test_loader)

