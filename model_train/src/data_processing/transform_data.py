from pandas import DataFrame


def merge_data(stock_data: DataFrame, economic_data: DataFrame, news_data: DataFrame, save_path, start_date="2018-01-01", end_date="2024-12-31"):
    """
    주식 데이터와 여러 경제 지표 데이터를 병합합니다.

    Args:
        stock_data (DataFrame): 주식 데이터 (다중 레벨 인덱스).
        economic_data (DataFrame): 경제 지표 데이터.
        news_data (DataFrame): 뉴스 데이터.
        save_path (str): 병합된 데이터를 저장할 경로.

    Returns:
        DataFrame: 병합된 데이터프레임.
    """

    # Ensure unique dates in each dataframe
    stock_data = stock_data.drop_duplicates(subset='Date').set_index('Date')
    economic_data = economic_data.drop_duplicates(subset='Date').set_index('Date')
    news_data = news_data.drop_duplicates(subset='Date').set_index('Date')

    # Filter stock_data to the specified date range
    stock_data = stock_data.loc[start_date:end_date]

    # Initial forward fill for economic and news data
    economic_data.ffill(inplace=True)
    news_data.ffill(inplace=True)

    # Reindex economic and news data to match stock data index
    economic_data = economic_data.reindex(stock_data.index)
    news_data = news_data.reindex(stock_data.index)

    # Fill missing values by forward filling again
    economic_data.ffill(inplace=True)
    news_data.ffill(inplace=True)

    # Rename avg_score to news_sentiment_score
    news_data = news_data.rename(columns={"avg_score": "news_sentiment_score"})

    # Debug: Print the first few rows of each dataframe
    print("Stock Data Sample:\n", stock_data.head())
    print("Economic Data Sample:\n", economic_data.head())
    print("News Data Sample:\n", news_data.head())

    # 뉴스 데이터 컬럼 필터링
    news_data = news_data[["news_sentiment_score"]]

    # 모든 경제 지표 주식 데이터에 병합
    merged_data: DataFrame = stock_data.join(economic_data, how="inner", rsuffix='_econ')
    merged_data = merged_data.join(news_data, how="inner", rsuffix='_news')

    # 데이터 저장
    merged_data.to_csv(f"{save_path}/merged_data.csv", mode="w")
    return merged_data
