import torch.nn as nn


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_dim=512, num_classes=10, num_layers=4):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
