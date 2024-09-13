from torch.utils.data import Dataset
import numpy as np
import random
import torch


class CFDataset(Dataset):
    def __init__(self, data_file_list, label_list, transform=None):
        self.data_file_list = data_file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, idx):
        data = np.loadtxt(self.data_file_list[idx], delimiter='\t', dtype=float).T
        spectrum = np.expand_dims(data[1], axis=0)
        label = int(self.label_list[idx])
        if self.transform:
            spectrum = self.transform(spectrum)
        return spectrum, label

    def input_size(self):
        data = np.loadtxt(self.data_file_list[0], delimiter='\t', dtype=float).T
        return data.shape[1]


def snn_batch(df, batch_size: int = 512, pos_neg_ratio=0.5):
    neg_size = int(batch_size / (pos_neg_ratio + 1))
    pos_size = batch_size - neg_size

    # negative pair
    neg_df = df[df['label'] == 0]
    neg_df = neg_df.sample(n=neg_size)
    # positive pair
    pos_df = df[df['label'] == 1]
    pos_df = pos_df.sample(n=pos_size)

    sample1_list, sample2_list, label_list = [], [], []
    for neg_i, neg_row in neg_df.iterrows():
        sample_data = np.loadtxt(neg_row['p300_file'], delimiter='\t', dtype=float).T
        neg_sample_data = np.loadtxt(neg_row['r300_file'], delimiter='\t', dtype=float).T
        sample1_list.append(sample_data[1])
        sample2_list.append(neg_sample_data[1])
        label_list.append(float(neg_row['label']))

    for pos_i, pos_row in pos_df.iterrows():
        sample_data = np.loadtxt(pos_row['p300_file'], delimiter='\t', dtype=float).T
        pos_sample_data = np.loadtxt(pos_row['r300_file'], delimiter='\t', dtype=float).T
        sample1_list.append(sample_data[1])
        sample2_list.append(pos_sample_data[1])
        label_list.append(float(pos_row['label']))

    # shuffle
    combined = list(zip(sample1_list, sample2_list, label_list))
    random.shuffle(combined)
    sample1_list, sample2_list, label_list = zip(*combined)

    sample1_list = np.array(sample1_list)
    sample2_list = np.array(sample2_list)

    sample1_list = np.expand_dims(sample1_list, 1)
    sample2_list = np.expand_dims(sample2_list, 1)

    return torch.tensor(sample1_list), torch.tensor(sample2_list), torch.tensor(label_list)
