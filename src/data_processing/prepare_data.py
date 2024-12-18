import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(merged_data=None):
    if merged_data is None:
        # 경제 데이터 로드
        merged_data = pd.read_csv('data/processed/merged_data.csv', index_col=0)


    # 타겟 변수: 주식의 다음 날 수익률
    merged_data['Stock_Return'] = merged_data['Close'].pct_change().shift(-1)

    # 결측값 제거
    merged_data = merged_data.dropna()

    # 특성 변수와 타겟 변수 분리
    X = merged_data[['FEDFUNDS', 'CPIAUCNS']]
    y = merged_data['Stock_Return']

    # 데이터 분할 (훈련 데이터와 테스트 데이터로)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 특성 데이터 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
