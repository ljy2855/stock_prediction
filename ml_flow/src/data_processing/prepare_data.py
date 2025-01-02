import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


INPUT_VAR = ['FEDFUNDS', 'CPIAUCNS', 'Close', 'Volume', 'High', 'Low', 'Open', 'Price']

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
    데이터를 Transformer 모델에 맞게 시퀀스 형태로 변환. 타겟은 다음 forecast_steps 동안의 누적 주가 변동률 벡터로 설정.
    :param merged_data: 병합된 데이터 (DataFrame)
    :param n_steps: 입력 시퀀스 길이 (과거 n_steps 사용)
    :param forecast_steps: 예측할 기간 길이 (다음 forecast_steps 동안의 누적 변동률)
    :param batch_size: DataLoader의 배치 크기
    :return: PyTorch DataLoader (train_loader, test_loader)
    """
    if merged_data is None:
        # 경제 데이터 로드
        merged_data = pd.read_csv('data/processed/merged_data.csv', index_col=0)

    # 타겟 변수: 다음 forecast_steps 동안의 누적 변동률
    price = merged_data['Price'].values
    target_momentum = [
        (price[i + 1:i + 1 + forecast_steps] - price[i]) / price[i]
        if i + 1 + forecast_steps <= len(price) else np.zeros(forecast_steps)
        for i in range(len(price))
    ]
    merged_data['Target_Momentum'] = target_momentum

    # 결측값 제거
    merged_data = merged_data.dropna()

    # 특성 변수와 타겟 변수 분리
    X = merged_data[INPUT_VAR].values
    y = np.vstack(merged_data['Target_Momentum'].values)  # 벡터 형태로 타겟 변환

    # 데이터 정규화 (Standard Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    # 시퀀스 데이터 생성 함수
    def create_sequences(X, y, n_steps):
        sequences_X, sequences_y = [], []
        for i in range(len(X) - n_steps):
            seq_X = X[i:i + n_steps]
            seq_y = y[i + n_steps - 1]  # 시퀀스의 마지막 지점에서 타겟 시작
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        return np.array(sequences_X), np.array(sequences_y)

    # 시퀀스 데이터 생성
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, n_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, n_steps)

    # PyTorch Dataset 객체 생성
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Dataset과 DataLoader 생성
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prepare_inference_input(n_steps, data_path=None):
    """
    실시간 데이터를 모델 입력 형식으로 변환합니다.
    :param raw_data: 실시간 주가 데이터 (DataFrame)
    :param n_steps: 입력 시퀀스 길이
    :return: 모델 입력 데이터 (Tensor)
    """
    if data_path is None:
        # 경제 데이터 로드
        data_path = "data/processed/merged_data.csv"
    raw_data = pd.read_csv(data_path, index_col=0)
    # 특성 변수 추출
    X = raw_data[INPUT_VAR].values

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 시퀀스 데이터 생성
    if len(X_scaled) < n_steps:
        raise ValueError(f"데이터의 길이가 n_steps({n_steps})보다 적습니다.")
    input_seq = X_scaled[-n_steps:]  # 최근 n_steps 시퀀스 사용
    return torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # (1, n_steps, input_size)

def prepare_backtest_input(data_path='data/processed/merged_data.csv', start_date=None, duration=None):
    """
    백테스트를 위한 데이터를 준비합니다.
    :param data_path: 데이터 경로
    :param start_date: 백테스트 시작 날짜
    :param duration: 백테스트 기간 (일 단위)
    :return: 가격 데이터, 특성 데이터, 날짜 데이터
    """
    # 경제 데이터 로드
    test_data = pd.read_csv(data_path)

    if test_data is None or test_data.empty:
        raise ValueError("데이터가 존재하지 않습니다.")

    # 날짜 정렬
    test_data["Date"] = pd.to_datetime(test_data["Date"])
    test_data = test_data.sort_values(by="Date")


    # 기간 필터링
    if start_date is None or duration is None:
        duration = 365
        random_start_idx = np.random.randint(0, len(test_data) - duration)  # 시작점 
        test_data = test_data.iloc[random_start_idx:random_start_idx + duration]  
    else:
        start_date = pd.to_datetime(start_date)
        test_data = test_data[test_data['Date'] >= start_date]
        test_data = test_data[test_data['Date'] < start_date + pd.Timedelta(days=duration)]

    if test_data.empty:
        raise ValueError("선택된 기간에 해당하는 데이터가 없습니다.")

    # 필요한 데이터 추출
    price_data = test_data["Price"].values  # 종가 데이터
    feature_data = test_data[INPUT_VAR].values  # 특성 데이터
    date_data = test_data["Date"].values  # 날짜 데이터

    print(f"선택된 데이터 기간: {test_data['Date'].iloc[0]} ~ {test_data['Date'].iloc[-1]}")
    return price_data, feature_data, date_data
