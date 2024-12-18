from datetime import datetime

from data_processing.prepare_data import prepare_data_for_sequences
from data_processing.stock_data import download_stock_data
from data_processing.fred_data import download_economic_data
from data_processing.transform_data import merge_data
from model.base import TimeSeriesModel
from model.lstm import LSTMModel
from model.transformer import TransformerTimeSeriesModel
from utils.config import config

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
    train_loader, test_loader = prepare_data_for_sequences(n_steps=30, batch_size=64)

    # LSTM 모델 초기화
    # input_size = 3
    # hidden_size = 64
    # output_size = 1

    # lstm_model = LSTMModel(input_size, hidden_size, output_size)
    # trainer = TimeSeriesModel(lstm_model, lr=0.001, model_name="LSTM")



    # Transformer 모델 초기화
    input_size = 3
    d_model = 64
    nhead = 4
    num_layers = 4
    output_size = 1

    transformer_model = TransformerTimeSeriesModel(input_size, d_model, nhead, num_layers, output_size)
    trainer = TimeSeriesModel(transformer_model, lr=0.001, model_name="Transformer")

    # 훈련 및 평가
    trainer.train(train_loader, num_epochs=50)
    trainer.evaluate(test_loader)
    trainer.save_model()
