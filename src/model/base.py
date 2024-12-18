import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import os

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, num_epochs=50, verbose=True):
        """
        모델 훈련 함수
        :param train_loader: PyTorch DataLoader 객체 (훈련 데이터)
        :param num_epochs: 학습 반복 횟수
        :param verbose: 학습 로그 출력 여부
        """
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader):
        """
        모델 평가 함수
        :param test_loader: PyTorch DataLoader 객체 (테스트 데이터)
        :return: MSE와 R² 점수
        """
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                y_pred.extend(outputs.squeeze().cpu().numpy())
                y_true.extend(targets.cpu().numpy())
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        return mse, r2

    def save_model(self, save_path="models"):
        """
        모델 저장 함수
        :param save_path: 모델 파일 저장 경로
        """
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{save_path}/{self.model_name}.pth")
        print(f"Model saved to {save_path}/{self.model_name}.pth")

    def load_model(self, save_path="models"):
        """
        모델 로드 함수
        :param save_path: 모델 파일 경로
        """
        model_path = f"{save_path}/{self.model_name}.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
