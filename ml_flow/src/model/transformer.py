import torch
import torch.nn as nn


class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, forecast_steps, dim_feedforward=2048, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.input_layer = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, forecast_steps)  # forecast_steps 크기의 벡터 출력
        self.to(self.device)  # 모델을 선택한 장치로 이동

    def forward(self, x):
        x = self.input_layer(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # 마지막 타임스텝 사용
        x = self.output_layer(x)  # (batch_size, forecast_steps)
        
        return x
