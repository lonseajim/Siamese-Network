import torch
import torch.nn as nn
from model.resnet import ResNet
from torch.autograd import Variable


class SNN(nn.Module):
    """
    Siamese Neural Network
    :param input1_dim: dim of spectrum input1
    :param input2_dim: dim of spectrum input2
    :param input_dim: dim of input for spectral encoder
    :param feature_dim: dim of spectral feature
    :param backbone: backbone network('transformer' or 'resnet' or 'lstm')
    :param r_num_layers: number of residual layers
    :param t_num_layers: number of transformer encoder layers
    :param l_num_layers: number of LSTM layers
    :param l_hidden_dim: hidden dim of LSTM layers
    """

    def __init__(self, input1_dim, input2_dim, input_dim=512, feature_dim=256, backbone='transformer',
                 r_num_layers=6, t_num_layers=3, l_num_layers=2, l_hidden_dim=100):
        super(SNN, self).__init__()

        self.input_dim = input_dim
        self.backbone = backbone

        # transform the input to the same dimension
        self.linear1 = nn.Linear(input1_dim, input_dim)
        self.linear2 = nn.Linear(input2_dim, input_dim)

        # transformer encoder
        t_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=input_dim * 4)
        self.t_encoder = nn.TransformerEncoder(t_encoder_layer, num_layers=t_num_layers)
        # features extracted by transformer mapped to the same dimension
        self.t_fc = nn.Linear(input_dim, feature_dim)

        # resnet encoder
        self.r_model = ResNet(hidden_sizes=[100] * r_num_layers, num_blocks=[2] * r_num_layers, input_dim=input_dim)
        r_z_dim = self._get_encoding_size()
        # features extracted by resnet mapped to the same dimension
        self.r_fc = nn.Linear(r_z_dim, feature_dim)

        # lstm encoder
        self.l_encoder = nn.LSTM(input_size=input_dim, hidden_size=l_hidden_dim, num_layers=l_num_layers,
                                 batch_first=True,
                                 bidirectional=True)
        # features extracted by lstm mapped to the same dimension
        self.l_fc = nn.Linear(2 * l_hidden_dim, feature_dim)

        self.sub_sim = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def transformer_encoder(self, input):
        x = self.t_encoder(input)
        return self.t_fc(x)

    def resnet_encoder(self, input):
        x = self.r_model.encode(input)
        return self.r_fc(x)

    def lstm_encoder(self, input):
        x, _ = self.l_encoder(input)
        return self.l_fc(x)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(1, 1, self.input_dim))
        z = self.r_model.encode(temp)
        z_dim = z.data.size(1)
        return z_dim

    def forward(self, input1, input2):
        input1 = self.linear1(input1)
        input2 = self.linear2(input2)

        if self.backbone == 'transformer':
            input1_features = self.transformer_encoder(input1)
            input2_features = self.transformer_encoder(input2)
        elif self.backbone == 'resnet':
            input1_features = self.resnet_encoder(input1)
            input2_features = self.resnet_encoder(input2)
        else:  # 'lstm':
            input1_features = self.lstm_encoder(input1)
            input2_features = self.lstm_encoder(input2)

        sim = self.sub_sim(torch.abs(input1_features - input2_features))
        return sim.flatten()


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        dist = nn.functional.pairwise_distance(x1, x2)
        total_loss = (1 - y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp_min_(self.margin - dist, 0), 2)
        loss = torch.mean(total_loss)
        return loss


if __name__ == '__main__':
    # test network
    backbone_list = ['transformer', 'resnet', 'lstm']
    input1_dim, input2_dim = 701, 701
    input1 = Variable(torch.rand(5, 1, input1_dim))
    input2 = Variable(torch.rand(5, 1, input2_dim))
    labels = torch.tensor([1., 0., 1., 0., 0.])

    model = SNN(input1_dim, input2_dim, backbone=backbone_list[0])
    out = model(input1, input2)
    print(out)
    criterion = nn.BCELoss()
    loss = criterion(out, labels)
    print(loss)
