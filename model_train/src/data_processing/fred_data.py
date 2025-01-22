import pandas as pd
from pandas_datareader import data as pdr
from fredapi import Fred
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.config import config

# FRED API 키
API_KEY = config.get_secret("FRED_API_KEY")
fred = Fred(api_key=API_KEY)

# 경제 지표 리스트
economic_indicators = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCNS": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
    "GDPC1": "Real GDP",
    "UMCSENT": "Consumer Confidence Index",
    "DGS10": "10-Year Treasury Rate",
    "HOUST": "Housing Starts",
    "ISRATIO": "Business Inventory to Sales Ratio",
    "TWEXB": "Trade Weighted U.S. Dollar Index",
    "PI": "Personal Income"
}

def download_economic_data(start_date, end_date, save_path):
    """
    지정된 경제 지표 데이터를 다운로드하고 저장합니다.
    """
    all_data = {}

    for code, description in economic_indicators.items():
        print(f"Downloading {description} ({code})...")
        try:
            # FRED 데이터 가져오기
            data = fred.get_series(code, start_date, end_date)
            data = pd.DataFrame(data, columns=[code])  # DataFrame으로 변환
            data.index.name = "Date"  # 인덱스를 Date로 명명

            # CSV 파일로 저장 (인덱스 포함)
            file_path = f"{save_path}/{code}_{start_date}_{end_date}.csv"
            data.to_csv(file_path, mode='w', index=True)  # index=True 추가
            print(f"Saved {description} to {file_path}")

            all_data[code] = data
        except Exception as e:
            print(f"Failed to download {description} ({code}): {e}")

    return all_data



if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    save_path = "/tmp"

    all_data = download_economic_data(start_date, end_date, save_path)

    for code, data in all_data.items():
        print(f"Last 5 entries for {code}:")
        print(data.tail())
