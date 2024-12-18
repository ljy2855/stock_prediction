import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

def prepare_data_for_sequences(merged_data=None, n_steps=30, batch_size=64):
    """
    데이터를 LSTM 모델에 맞게 시퀀스 형태로 변환.
    :param merged_data: 병합된 데이터 (DataFrame)
    :param n_steps: 입력 시퀀스 길이 (과거 n_steps 사용)
    :param batch_size: DataLoader의 배치 크기
    :return: PyTorch DataLoader (train_loader, test_loader)
    """
    if merged_data is None:
        # 경제 데이터 로드
        merged_data = pd.read_csv('data/processed/merged_data.csv', index_col=0)

    # 타겟 변수: 주식의 다음 날 수익률
    merged_data['Stock_Return'] = merged_data['Close'].pct_change().shift(-1)

    # 결측값 제거
    merged_data = merged_data.dropna()

    # 특성 변수와 타겟 변수 분리
    X = merged_data[['FEDFUNDS', 'CPIAUCNS', 'Close']].values
    y = merged_data['Stock_Return'].values

    # 데이터 정규화 (Standard Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # 시퀀스 데이터 생성 함수
    def create_sequences(X, y, n_steps):
        sequences_X, sequences_y = [], []
        for i in range(len(X) - n_steps):
            seq_X = X[i:i+n_steps]
            seq_y = y[i+n_steps]
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
