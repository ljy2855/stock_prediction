import torch
import torch.nn as nn
import os

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward=512, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_layer(x)  # 임베딩
        x = x.permute(1, 0, 2)  # Transformer 입력 형태로 변환 (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)  # Transformer 인코더
        x = x[-1]  # 마지막 타임스텝 사용
        x = self.output_layer(x)  # 출력층
        return x