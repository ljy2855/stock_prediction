import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention Mechanism  
    입력된 Transformer 출력을 기반으로 각 시퀀스 단계별 가중치를 조정합니다.
    """
    def __init__(self, d_model):
        super(AdaptiveAttention, self).__init__()
        self.attention_weights = nn.Linear(d_model, 1)  # 각 시퀀스 단계별 가중치를 계산

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        weights = torch.sigmoid(self.attention_weights(x))  # (batch_size, seq_len, 1)
        attended_x = x * weights  # (batch_size, seq_len, d_model)
        return attended_x


class TransformerTimeSeriesModel(nn.Module):
    """
    Transformer 기반 시계열 예측 모델  
    """
    def __init__(
        self, input_size, d_model, nhead, num_layers, forecast_steps, dim_feedforward=2048, dropout=0.1
    ):
        super(TransformerTimeSeriesModel, self).__init__()
        self.device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        # ✅ 입력 레이어: 입력 데이터를 d_model 차원으로 변환
        self.input_layer = nn.Linear(input_size, d_model)

        # ✅ Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # ✅ Adaptive Attention Mechanism
        self.adaptive_attention = AdaptiveAttention(d_model)

        # ✅ 출력 레이어: 최종 예측을 forecast_steps 차원으로 변환
        self.output_layer = nn.Linear(d_model, forecast_steps)

        # ✅ 모델을 디바이스로 이동
        self.to(self.device)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        x = self.input_layer(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.adaptive_attention(x)  # (batch_size, seq_len, d_model)
        
        # ✅ 마지막 시퀀스 단계 선택
        x = x[:, -1, :]  # (batch_size, d_model)
        x = self.output_layer(x)  # (batch_size, forecast_steps)
        return x


# ✅ 손실 함수: MSELoss로 변경
class MSELoss(nn.Module):
    """
    Mean Squared Error Loss  
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


# ✅ 방향 정확도 계산 함수
def directional_accuracy(y_true, y_pred):
    """
    방향 정확도(Direction Accuracy)를 계산합니다.
    :param y_true: 실제 값
    :param y_pred: 예측 값
    :return: 방향 정확도
    """
    correct_directions = torch.sign(y_true) == torch.sign(y_pred)
    accuracy = torch.mean(correct_directions.float())
    return accuracy
