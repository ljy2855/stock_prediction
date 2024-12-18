from datetime import datetime

from data_processing.prepare_data import prepare_data
from data_processing.stock_data import download_stock_data
from data_processing.fred_data import download_economic_data
from data_processing.transform import merge_data
from model.evaluation import evaluate_model
from model.tranning import train_and_save_model
from utils.config import config

START_DATE = "2015-01-01"
TICKER = "SPY"
END_DATE = datetime.today().strftime('%Y-%m-%d')


if __name__ == "__main__":
    
    # 경로 설정
    stock_data_path = config.get_path("paths", "raw_data", "stock_data")
    
    # 데이터 다운로드 및 저장
    stock_data = download_stock_data(TICKER, START_DATE, END_DATE, stock_data_path)

    # 경제 데이터 다운로드 및 저장
    rate_data, cpi_data = download_economic_data(START_DATE, END_DATE, config.get_path("paths", "raw_data", "economic_data"))

    # 데이터 병합
    merged_data = merge_data(stock_data, rate_data, cpi_data, config.get_path("paths", "processed_data", "stock_data"))

    X_train, X_test, y_train, y_test = prepare_data()

    # 3. 모델 훈련 및 저장
    model_path = "./models/stock_model.pkl"
    model = train_and_save_model(X_train, X_test, y_train, y_test, model_path)

    # 4. 모델 평가
    evaluate_model(model_path, X_test, y_test)
