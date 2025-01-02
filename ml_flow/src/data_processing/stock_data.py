import yfinance as yf
import pandas as pd



def download_stock_data(ticker, start_date, end_date, save_path):
    """
    주식 시계열 데이터를 다운로드하고 저장
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.to_csv(f"{save_path}/{ticker}_stock_data_{start_date}_{end_date}.csv", mode='w')
    print(f"Data saved to {save_path}/{ticker}_stock_data_{start_date}_{end_date}.csv")

    return stock_data

