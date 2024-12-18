
def merge_data(stock_data, rate_data, cpi_data,save_path):
    # 주식 데이터와 경제 지표 데이터를 병합 (inner join)
    rate_data = rate_data.reindex(stock_data.index, method='ffill')  # NaN을 앞선 값으로 채움
    cpi_data = cpi_data.reindex(stock_data.index, method='ffill')  # NaN을 앞선 값으로 채움

    # 데이터 저장
    merged_data = stock_data.join([rate_data, cpi_data], how='inner')
    merged_data.columns = [
        'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume', 
        'FEDFUNDS', 'CPIAUCNS'
    ]

    merged_data.to_csv(f"{save_path}/merged_data.csv", mode='w')
    return merged_data