

import pandas as pd
from pandas_datareader import data as pdr
from fredapi import Fred

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.config import config

# FRED API 키
API_KEY = config.get_secret("FRED_API_KEY")
fred = Fred(api_key=API_KEY)

def download_economic_data(start_date, end_date,save_path):
    # 기준금리 데이터 (FEDFUNDS)
    rate_data = fred.get_series('FEDFUNDS', start_date, end_date)
    rate_data = pd.DataFrame(rate_data, columns=['FEDFUNDS'])
    
    # 물가 상승률 (CPI)
    cpi_data = pdr.DataReader('CPIAUCNS', 'fred', start_date, end_date)
    cpi_data = pd.DataFrame(cpi_data, columns=['CPIAUCNS'])

    # 데이터 저장
    rate_data.to_csv(f"{save_path}/rate_data_{start_date}_{end_date}.csv", mode='w')
    cpi_data.to_csv(f"{save_path}/cpi_data_{start_date}_{end_date}.csv", mode='w')

    return rate_data, cpi_data


if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    save_path = "/tmp"

    rate_data, cpi_data = download_economic_data(start_date, end_date, save_path)
    print(rate_data.tail())
    print(cpi_data.tail())