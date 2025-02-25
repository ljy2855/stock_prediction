from datetime import datetime

import pandas as pd

from data_processing.prepare_data import prepare_data_for_sequences
from data_processing.stock_data import download_stock_data
from data_processing.fred_data import download_economic_data
from data_processing.news_data import get_news_semantic_score
from data_processing.transform_data import merge_data
from model.base import TimeSeriesModel
from model.lstm import LSTMModel
from model.transformer import TransformerTimeSeriesModel
from utils.config import config

START_DATE = "2018-01-01"
TICKER = "SPY"
END_DATE = "2024-12-31"


def downlaod_data():
    # 경로 설정
    stock_data_path = config.get_path("paths", "raw_data", "stock_data")
    news_sentiment_data_path = config.get_path("paths", "processed_data", "news_sentiment")
    economic_data_path = config.get_path("paths", "raw_data", "economic_data")
    
    # 데이터 다운로드 및 저장
    stock_data = download_stock_data(TICKER, START_DATE, END_DATE, stock_data_path)

    # 경제 데이터 다운로드 및 저장
    economic_data = download_economic_data(START_DATE, END_DATE, economic_data_path)

    # 뉴스 데이터 다운로드 및 저장
    news_data = get_news_semantic_score(START_DATE, END_DATE, news_sentiment_data_path)


def load_data():
    processed_data_path = config.get_path("paths", "processed_data", "stock_data")
    stock_data_path = config.get_path("paths", "raw_data", "stock_data")
    economic_data_path = config.get_path("paths", "raw_data", "economic_data")
    news_sentiment_data_path = config.get_path("paths", "processed_data", "news_sentiment")

    stock_data = pd.read_csv(stock_data_path+ "/SPY_stock_data_2015-01-01_2025-01-17.csv")
    economic_data = pd.read_csv(economic_data_path+ "/economic_data_2018-01-01_2024-12-31.csv")
    news_data = pd.read_csv(news_sentiment_data_path)

    merged_data = merge_data(stock_data, economic_data, news_data, processed_data_path)

    return stock_data, economic_data, news_data


if __name__ == "__main__":
    load_data()