import torch
import torch.nn as nn

import torch
import torch.nn as nn

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, forecast_steps, dim_feedforward=2048, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, forecast_steps)  # forecast_steps 크기의 벡터 출력

    def forward(self, x):
        x = self.input_layer(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # 마지막 타임스텝 사용
        x = self.output_layer(x)  # (batch_size, forecast_steps)
        return x


    def inference(self, sequence):
        """
        새로운 시퀀스에 대한 Inference 수행.
        :param sequence: (seq_len, input_size) 형태의 입력 시퀀스 (단일 샘플)
        :return: 모델 예측값 (float)
        """
        self.eval()  # 평가 모드 설정
        with torch.no_grad():
            if len(sequence.shape) == 2:
                sequence = sequence.unsqueeze(0)  # 배치 차원 추가: (1, seq_len, input_size)
            prediction = self(sequence)  # (1, output_size)
            return prediction.squeeze().item()  # 스칼라 값 반환
