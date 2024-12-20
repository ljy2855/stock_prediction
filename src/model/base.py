from uuid import uuid4
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from tqdm import tqdm
import numpy as np

class TimeSeriesModel:
    def __init__(self, model, lr=0.001, model_name="Model"):
        """
        TimeSeriesModel 클래스를 초기화합니다.
        :param model: PyTorch 모델 객체
        :param input_size: 입력 특성 크기
        :param output_size: 출력 크기
        :param lr: 학습률
        :param model_name: 모델 이름 (저장 파일명)
        """
        self.model_name = model_name
        self.id = uuid4().hex
        self.device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, num_epochs=50, verbose=True, early_stopping=False, patience=10):
        """
        모델 훈련 함수
        :param train_loader: PyTorch DataLoader 객체 (훈련 데이터)
        :param num_epochs: 학습 반복 횟수
        :param verbose: 학습 로그 출력 여부
        :param early_stopping: 조기 종료 여부
        :param patience: 조기 종료를 위한 허용 에포크 수
        :return: 학습 손실 기록
        """
        self.model.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        loss_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward 및 Backward
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)  # (batch_size, forecast_steps)
                    loss = self.criterion(outputs, targets)  # MSE 또는 CrossEntropy
                    loss.backward()
                    self.optimizer.step()

                    # 배치 손실 기록 및 출력
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    tepoch.set_postfix(batch_loss=batch_loss)

            # 에포크 평균 손실 계산
            epoch_loss /= len(train_loader)
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

            # 조기 종료 로직
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

    def evaluate(self, test_loader, report_path="report", additional_info=None):
        """
        모델 평가 함수 (회귀 문제에 적합, MSE 기반)
        :param test_loader: PyTorch DataLoader 객체 (테스트 데이터)
        :param report_path: 평가 결과를 저장할 폴더 경로
        :param additional_info: 추가 정보 (파라미터, 배치 크기 등) 딕셔너리
        :return: MSE, R², Adjusted R² 점수
        """
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)  # 모델 예측
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

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

        # 보고서 파일 경로
        os.makedirs(report_path, exist_ok=True)
        report_file = os.path.join(report_path, "evaluation_results.json")

        # 모델 파라미터 가져오기
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 새 평가 결과
        new_entry = {
            "model_name": self.model_name,
            "model_id": self.id,
            "evaluation_metrics": {
                "mean_squared_error": mse,
                "r2_score": r2,
                "adjusted_r2_score": adjusted_r2,
            },
            "model_parameters": {
                "total_params": model_params,
                "layer_details": self.get_model_details(),
            },
            "additional_info": additional_info or {},
        }

        # 기존 보고서 읽기 (없으면 빈 리스트 생성)
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # 새 데이터를 기존 데이터에 추가
        existing_data.append(new_entry)

        # 업데이트된 데이터를 저장
        with open(report_file, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"Evaluation report updated at {report_file}")
        return mse, r2, adjusted_r2


    def save_model(self, save_path="models"):
        """
        모델 저장 함수
        :param save_path: 모델 파일 저장 경로
        """
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{save_path}/{self.model_name}_{self.id}.pth")
        print(f"Model saved to {save_path}/{self.model_name}.pth")
