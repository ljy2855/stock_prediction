import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader



INPUT_VAR = [
    'FEDFUNDS',        # Federal Funds Rate
    'Close',           # 주식 가격 (대표 값, 종가 또는 특정 값)
    'CPIAUCNS',        # Consumer Price Index
    'Volume',          # 거래량
    'UNRATE',          # Unemployment Rate
    'GDPC1',           # Real GDP
    'UMCSENT',         # Consumer Confidence Index
    'DGS10',           # 10-Year Treasury Rate
    'HOUST',           # Housing Starts
    'ISRATIO',         # Inventory to Sales Ratio
    'TWEXB',           # Trade Weighted U.S. Dollar Index
    'PI',              # Personal Income
]

SCALED_VAR = [
    'Close',           # 주식 가격 (값의 크기가 큼)
    'Volume',          # 거래량 (값의 크기가 큼)
    'GDPC1',           # Real GDP (값의 크기가 큼)
    'CPIAUCNS',        # Consumer Price Index (물가 수준)
    'PI',              # Personal Income (값의 크기가 큼)
    'HOUST',           # Housing Starts (값의 크기가 큼)
    'TWEXB',           # Trade Weighted U.S. Dollar Index (변동성 있음)
]


def create_labels(price_data, window=20, threshold=0.01):
    """
    앞으로의 한 달간 주가 모멘텀을 기준으로 라벨 생성.
    :param price_data: 종가 데이터 (numpy 배열 또는 pandas Series)
    :param window: 한 달간의 기간 (데이터 포인트 개수)
    :param threshold: 상승/하락을 판단하는 변화율 임계값
    :return: 범주형 라벨 (0: 상승, 1: 하락, 2: 정체)
    """
    labels = []
    for i in range(len(price_data)):
        if i + window < len(price_data):
            # 한 달간의 평균 변화율 계산
            future_window = price_data[i:i+window]
            future_return = (future_window[-1] - price_data[i]) / price_data[i]
            
            # 상승/하락/정체 분류
            if future_return > threshold:
                labels.append(0)  # 상승
            elif future_return < -threshold:
                labels.append(1)  # 하락
            else:
                labels.append(2)  # 정체
        else:
            labels.append(2)  # 데이터가 부족한 경우 정체로 처리
    return labels


def prepare_data_for_sequences(merged_data=None, n_steps=30, forecast_steps=30, batch_size=64):
    """
    데이터를 Transformer 모델에 맞게 시퀀스 형태로 변환.
    특정 피처(FEDFUNDS, CPIAUCNS)는 스케일링에서 제외하며, 나머지 피처는 로그 변화율로 변환.
    """
    if merged_data is None:
        merged_data = pd.read_csv('data/processed/merged_data.csv', index_col=0)
    
    # Target Momentum 생성
    price = merged_data['Close'].values
    rolling_window = 10  # 이동 평균 창 크기
    price_smooth = pd.Series(price).rolling(rolling_window).mean().values
    target_momentum = np.zeros(len(price))
    for i in range(len(price) - forecast_steps):
        target_momentum[i] = (price_smooth[i + forecast_steps] - price_smooth[i]) / price_smooth[i]
    merged_data['Target_Momentum'] = target_momentum

    
    # 피처 로그 변화율로 변경
    scale_features = SCALED_VAR  # 변화율로 변환할 피처
    for feature in scale_features:
        merged_data[feature] = merged_data[feature].pct_change()

    # 결측치 제거 (모든 연산 후)
    merged_data = merged_data.dropna().reset_index(drop=True)
    # 특성 및 타겟 분리
    X = merged_data[INPUT_VAR].values  # 입력 피처 데이터
    y = merged_data['Target_Momentum'].values.reshape(-1, 1)  # 타겟 데이터

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    if len(X_test) < n_steps:
        raise ValueError("Not enough test data to create sequences. Increase test size or reduce n_steps.")
    
    # 시퀀스 데이터 생성 함수
    def create_sequences(X, y, n_steps):
        sequences_X, sequences_y = [], []
        seq_length = min(len(X), len(y)) - n_steps
        for i in range(seq_length):
            seq_X = X[i:i + n_steps]
            seq_y = y[i + n_steps - 1]
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        return np.array(sequences_X), np.array(sequences_y)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, n_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, n_steps)
    print(f"Train Sequence Shape: {X_train_seq.shape}, Test Sequence Shape: {X_test_seq.shape}")

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # DataLoader 생성
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def prepare_backtest_input(data_path='data/processed/merged_data.csv', start_date=None, duration=365):
    """
    백테스트 데이터를 준비합니다. 가격, 특성, 날짜만 반환합니다.
    """
    # 데이터 로드
    test_data = pd.read_csv(data_path)

    if test_data is None or test_data.empty:
        raise ValueError("데이터가 존재하지 않습니다.")

    # 날짜 정렬
    test_data["Date"] = pd.to_datetime(test_data["Date"])
    test_data = test_data.sort_values(by="Date")

    # 기간 필터링
    if start_date is None:
        random_start_idx = np.random.randint(0, len(test_data) - duration)
        test_data = test_data.iloc[random_start_idx:random_start_idx + duration]
    else:
        start_date = pd.to_datetime(start_date)
        test_data = test_data[test_data['Date'] >= start_date]
        test_data = test_data[test_data['Date'] < start_date + pd.Timedelta(days=duration)]

    if test_data.empty:
        raise ValueError("선택된 기간에 해당하는 데이터가 없습니다.")

    # 원본 Close 값 저장
    original_price_data = test_data['Close'].values

    # 로그 변화율 적용
    scale_features = SCALED_VAR
    for feature in scale_features:
        test_data[feature] = test_data[feature].pct_change()

    # 결측치 제거
    test_data = test_data.dropna().reset_index(drop=True)
    original_price_data = original_price_data[len(original_price_data) - len(test_data):]

    # 입력 특성 준비
    feature_data = test_data[INPUT_VAR].values  # 특성 데이터
    date_data = test_data['Date'].values        # 결측치 제거된 날짜 데이터

    print(f"Feature Shape: {feature_data.shape}, Price Shape: {original_price_data.shape}, Date Shape: {len(date_data)}")

    return original_price_data, feature_data, date_data




def prepare_transformer_input(feature_data, n_steps=30):
    """
    Transformer 입력으로 사용할 수 있도록 데이터를 시퀀스로 변환합니다.
    :param feature_data: 특성 데이터 (numpy.ndarray)
    :param n_steps: 시퀀스 길이
    :return: Transformer 입력 데이터 (torch.Tensor)
    """
    sequences = []
    for i in range(len(feature_data) - n_steps + 1):
        seq = feature_data[i:i + n_steps]  # 30일 분량 추출
        sequences.append(seq)
    
    sequences = np.array(sequences)  # (batch_size, seq_len, feature_dim)
    input_tensor = torch.tensor(sequences, dtype=torch.float32)
    
    print(f"Transformer Input Shape: {input_tensor.shape}")  # (batch_size, seq_len, feature_dim)
    return input_tensor

if __name__ == "__main__":
    train_loader, test_loader = prepare_data_for_sequences(n_steps=30, batch_size=30,forecast_steps=30)
    for X, y in train_loader:
        print(f"Train Input Shape: {X.shape}, Train Target Shape: {y.shape}")
        break

    for X, y in test_loader:
        print(f"Test Input Shape: {X.shape}, Test Target Shape: {y.shape}")
        break

    price_data, feature_data, date_data = prepare_backtest_input(duration=180)
    print(f"Price Data Shape: {price_data.shape}, Feature Data Shape: {feature_data.shape}, Date Data Shape: {date_data.shape}")
