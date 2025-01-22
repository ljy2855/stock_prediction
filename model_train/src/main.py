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
    economic_data = download_economic_data(START_DATE, END_DATE, config.get_path("paths", "raw_data", "economic_data"))

    # 데이터 병합
    merged_data = merge_data(stock_data, economic_data, config.get_path("paths", "processed_data", "stock_data"))

if __name__ == "__main__":
    init_data()