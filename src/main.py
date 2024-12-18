from datetime import datetime

from data_processing.prepare_data import prepare_data_for_sequences
from data_processing.stock_data import download_stock_data
from data_processing.fred_data import download_economic_data
from data_processing.transform import merge_data
from model.base import LSTMModel, TimeSeriesModel
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

    # 모델 훈련 및 저장
    model = TimeSeriesModel(LSTMModel, input_size=2, hidden_size=64, output_size=1, model_name="LSTM")
    model.train(train_loader, num_epochs=50)
    model.evaluate(test_loader)
    model.save_model()
