import yfinance as yf
import pandas as pd


def download_stock_data(ticker, start_date, end_date, save_path):
    """
    주식 시계열 데이터를 다운로드하고 저장합니다.

    Args:
        ticker (str): 주식 종목 코드.
        start_date (str): 시작 날짜 (YYYY-MM-DD 형식).
        end_date (str): 종료 날짜 (YYYY-MM-DD 형식).
        save_path (str): 데이터를 저장할 경로.

    Returns:
        DataFrame: 다운로드된 주식 데이터.
    """
    # 주식 데이터 다운로드
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # 인덱스를 컬럼으로 변환
    stock_data.reset_index(inplace=True)

    # 첫 번째 행에 잘못된 컬럼 레벨(SPY 등)이 포함되었을 경우 제거
    if stock_data.columns.nlevels > 1:
        print("Removing unnecessary column levels...")
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Date 컬럼을 datetime 타입으로 변환 후 인덱스로 설정
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data.set_index("Date", inplace=True)

    # CSV로 저장 (인덱스 포함)
    file_path = f"{save_path}/{ticker}_stock_data_{start_date}_{end_date}.csv"
    stock_data.to_csv(file_path, index=True)  # index=True로 인덱스 포함
    print(f"Data saved to {file_path}")

    return stock_data
