
from pandas import DataFrame


def merge_data(stock_data : DataFrame, economic_data_dict, save_path):
    """
    주식 데이터와 여러 경제 지표 데이터를 병합합니다.

    Args:
        stock_data (DataFrame): 주식 데이터 (다중 레벨 인덱스).
        economic_data_dict (dict): 경제 지표 데이터 딕셔너리 {지표명: DataFrame}.
        save_path (str): 병합된 데이터를 저장할 경로.

    Returns:
        DataFrame: 병합된 데이터프레임.
    """

    # 경제 지표 데이터 병합
    for indicator, data in economic_data_dict.items():
        # NaN 값을 앞선 값으로 채움
        economic_data_dict[indicator] = data.reindex(stock_data.index, method="ffill")

    # 모든 경제 지표를 주식 데이터에 병합
    merged_data : DataFrame = stock_data.copy()
    for indicator, data in economic_data_dict.items():
        merged_data = merged_data.join(data, how="inner")

    # 컬럼명 업데이트 (주식 데이터와 경제 지표 포함)
    stock_columns = list(stock_data.columns)
    indicator_columns = list(economic_data_dict.keys())
    column_names = stock_columns + indicator_columns

    merged_data.columns = column_names
    # 데이터 저장
    merged_data.to_csv(f"{save_path}/merged_data.csv", mode="w")
    return merged_data
