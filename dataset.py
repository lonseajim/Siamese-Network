import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from base import SpectraFolder


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


def create_folder_csv(folder_src, csv_path=None):
    """
    create a csv file with data folder
    """
    sf = SpectraFolder(folder_src)
    df = sf.get_df()
    if csv_path is not None:
        df.to_csv(csv_path)
    else:
        df.to_csv('dataset.csv')


def create_fsl_ds_csv(query_csv, support_csv, n_splits=5, n_export=5):
    """
    create csv files for few-shot learning training and testing
    """
    assert 0 < n_export <= n_splits

    query_df = pd.read_csv(query_csv, index_col=0)
    support_df = pd.read_csv(support_csv, index_col=0)

    query_file_list = query_df['data_file'].to_numpy()
    query_label_list = query_df['label'].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for idx, (train_index, test_index) in enumerate(skf.split(query_file_list, query_label_list)):
        if idx >= n_export:
            break

        train_query_df = query_df.iloc[test_index]
        train_support_df = support_df.iloc[test_index]

        test_query_df = query_df.iloc[train_index]
        test_support_df = support_df.iloc[train_index]

        few_query_len = len(test_index)
        test_query_len = len(train_index)

        # query dataset
        train_query_df.to_csv('train_P300_600_cv{}_{}.csv'.format(idx + 1, few_query_len))
        # support dataset
        train_support_df.to_csv('train_R300_1200_cv{}_{}.csv'.format(idx + 1, few_query_len))

        # for model testing
        test_query_df.to_csv('test_P300_600_cv{}_{}.csv'.format(idx + 1, test_query_len))
        test_support_df.to_csv('test_R300_1200_cv{}_{}.csv'.format(idx + 1, test_query_len))


def merge_fsl_csv(cv=1):
    """
    merge two csv files into one csv file for snn model training
    create pos and neg pairs for dataset
    """
    p300_df = pd.read_csv('train_P300_600_cv{}_100.csv'.format(cv))
    r300_df = pd.read_csv('train_R300_1200_cv{}_100.csv'.format(cv))
    data_list = []
    for p300_i, p300_row in p300_df.iterrows():
        p300_data_file = p300_row['data_file']
        p300_data_label = p300_row['label']
        for r300_i, r300_row in r300_df.iterrows():
            r300_data_file = r300_row['data_file']
            r300_data_label = r300_row['label']
            if p300_data_label == r300_data_label:
                label = 1
            else:
                label = 0
            data_list.append([p300_data_file, r300_data_file, label])
    df = pd.DataFrame(data_list, columns=['p300_file', 'r300_file', 'label'])
    df.to_csv('fsl2_cv{}_{}.csv'.format(cv, len(data_list)))
