import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import os

class TimeSeriesModel:
    def __init__(self, model, input_size, output_size, hidden_size=64, lr=0.001, model_name="LSTM"):
        """
        TimeSeriesModel 클래스를 초기화합니다.
        :param model: PyTorch 모델 클래스
        :param input_size: 입력 특성의 크기
        :param output_size: 출력 특성의 크기
        :param hidden_size: LSTM 등에서 사용하는 히든 레이어 크기
        :param lr: 학습률
        :param model_name: 모델의 이름 (저장/로드에 사용)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, num_epochs=50):
        """
        모델 훈련 함수
        :param train_loader: PyTorch DataLoader 객체 (훈련 데이터)
        :param num_epochs: 훈련 반복 횟수
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
        self.model.load_state_dict(torch.load(f"{save_path}/{self.model_name}.pth", map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {save_path}/{self.model_name}.pth")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :]) 
        return output

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :]) 
        return output
