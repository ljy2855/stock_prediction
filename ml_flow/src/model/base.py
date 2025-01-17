from datetime import datetime
from uuid import uuid4
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from tqdm import tqdm
import numpy as np
from .mlflow_logging import mlflow_log_evaluation, mlflow_log_training

class TimeSeriesModel:
    def __init__(self, model, lr=0.001, model_name="Model"):
        """
        TimeSeriesModel 클래스를 초기화합니다.
        :param model: PyTorch 모델 객체
        :param lr: 학습률
        :param model_name: 모델 이름 (저장 파일명)
        """
        self.model_name = model_name + datetime.now().strftime("_%Y%m%d_%H%M%S")
        self.device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    @mlflow_log_training
    def train(self, train_loader, num_epochs=50, verbose=True, early_stopping=False, patience=10):
        self.model.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        loss_history = []

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # 학습률 스케줄러 추가

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward 및 Backward
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    tepoch.set_postfix(batch_loss=loss.item())
                    
                    
                    # print(f"Sample prediction: {outputs[0].item()}, Target: {targets[0].item()}")
                    
                scheduler.step()  # 학습률 업데이트

            epoch_loss /= len(train_loader)
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

            # Early Stopping
            if early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break

        return loss_history



    def get_model_details(self):
        """
        모델의 레이어 세부 정보를 추출하여 JSON 형식으로 반환.
        :return: 모델의 레이어 정보 (리스트 형식)
        """
        model_details = []
        for name, layer in self.model.named_children():
            layer_info = {
                "layer_name": name,
                "layer_type": layer.__class__.__name__,
                "details": self.get_layer_details(layer)
            }
            model_details.append(layer_info)
        return model_details

    def get_layer_details(self, layer):
        """
        레이어 객체의 주요 속성을 추출하여 JSON-friendly 딕셔너리로 반환.
        :param layer: PyTorch 레이어 객체
        :return: 레이어 속성 (딕셔너리 형식)
        """
        layer_details = {}
        if hasattr(layer, 'in_features'):
            layer_details['in_features'] = layer.in_features
        if hasattr(layer, 'out_features'):
            layer_details['out_features'] = layer.out_features
        if hasattr(layer, 'kernel_size'):
            layer_details['kernel_size'] = layer.kernel_size
        if hasattr(layer, 'stride'):
            layer_details['stride'] = layer.stride
        if hasattr(layer, 'padding'):
            layer_details['padding'] = layer.padding
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer_details['bias'] = True
        else:
            layer_details['bias'] = False
        return layer_details

    @mlflow_log_evaluation
    def evaluate(self, test_loader, additional_info=None, window_size=30):
        """
        모델 평가 함수 (회귀 문제에 적합, MSE 기반)
        데이터를 30일 단위로 슬라이싱하여 평가.
        :param test_loader: PyTorch DataLoader 객체 (테스트 데이터)
        :param additional_info: 추가 정보 (파라미터, 배치 크기 등) 딕셔너리
        :param window_size: 슬라이싱할 창 크기 (기본값: 30일)
        :return: MSE, R², Adjusted R² 점수
        """
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 30일 단위로 슬라이싱
                batch_size, seq_len, num_features = inputs.shape
                for start_idx in range(0, seq_len - window_size + 1, window_size):
                    end_idx = start_idx + window_size
                    input_slice = inputs[:, start_idx:end_idx, :]  # 30일 단위 슬라이싱
                    target_slice = targets  # 전체 타겟 값

                    # 모델 예측
                    outputs = self.model(input_slice)
                    print(outputs.shape)
                    print(outputs.cpu().numpy())
                    y_pred.extend(outputs.cpu().numpy())
                    y_true.extend(target_slice.cpu().numpy())

        # NumPy 배열로 변환
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 성능 평가
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 조정된 R² 계산
        n = len(y_true)  # 샘플 수
        k = test_loader.dataset[0][0].shape[-1]  # 입력 변수(특성) 수
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

        # 평가 결과 출력
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Adjusted R² Score: {adjusted_r2:.4f}")

        return mse, r2, adjusted_r2

