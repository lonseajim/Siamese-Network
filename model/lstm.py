import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.LSTM(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True),
        )
        self.output = nn.Linear(2 * hidden_size, out_size)

    def forward(self, x):
        """
        :param x: shape(batch, seq_len, input_size)
        :return:
        """
        batch, seq_len, nums_fea = x.size()
        features, _ = self.features(x)
        # output = self.classifier(features)
        output = self.output(features.view(batch * seq_len, -1))
        return output
