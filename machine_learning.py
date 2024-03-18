import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import raman.plot as plot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold


def do_PCA(spectra, components=2):
    pca = PCA(n_components=components)
    pca.fit(spectra)
    return pca.transform(spectra)


def do_tSNE(spectra, components=2):
    tsne = TSNE(n_components=components, init='pca', random_state=1, method='exact')
    return tsne.fit_transform(spectra)


def do_KNN():
    x = np.array(list(range(100)))
    y = np.array(list(range(10)))
    y = np.tile(y, 10)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(x, y):
        print(train_index, test_index)


if __name__ == '__main__':
    p300_df = pd.read_csv('data/P300_1200g_norm.csv', index_col=0)
    r300_df = pd.read_csv('data/R300_1200g_norm.csv', index_col=0)

    p300_x, p300_y, r300_x, r300_y = [], [], [], []
    for idx, row in p300_df.iterrows():
        data = np.loadtxt(row['data_file'], delimiter='\t', dtype=float).T
        p300_x.append(data[1])
        p300_y.append(int(row['label']))

    for idx, row in r300_df.iterrows():
        data = np.loadtxt(row['data_file'], delimiter='\t', dtype=float).T
        r300_x.append(data[1])
        r300_y.append(int(row['label']))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    # 数据拆分器
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 准确率
    scores = []
    other_ds_scores = []
