import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import machine_learning as ml
from plot import color_pick, marker_pick

G_INS = ['P300', 'R300']
G_CS = ['ABA', 'CFR', 'ECO', 'EFA', 'KOX', 'KPN', 'PMA', 'SAU', 'SHO', 'SMA']


def plot_3D_scatter(data: np.array, labels: np.array, fig_size=(7, 5), fig_dpi=300, fig_title=None):
    assert len(data) == len(labels)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    ax = fig.add_subplot(111, projection='3d')
    for in_idx, ins in enumerate(G_INS):
        for c_idx, c in enumerate(G_CS):
            label = ins + '-' + c
            ax.scatter(xs=data[labels == label, 0], ys=data[labels == label, 1], zs=data[labels == label, 2],
                       c=color_pick(c_idx), marker=marker_pick(in_idx), s=2, label=label)

    plt.legend(bbox_to_anchor=(1.06, 0), fontsize='5', loc=3, borderaxespad=0)

    if fig_title is not None:
        plt.title(fig_title)


def plot_3D_scatter2(data: np.array, labels: np.array, fig_size=(7, 5), fig_dpi=300, fig_title=None):
    assert len(data) == len(labels)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    ax = fig.add_subplot(111, projection='3d')
    for in_idx, ins in enumerate(G_INS):
        # for c_idx, c in enumerate(G_CS):
        #     label = ins + '-' + c
        #     ax.scatter(xs=data[labels == label, 0], ys=data[labels == label, 1], zs=data[labels == label, 2],
        #                c=color_pick(in_idx), marker=marker_pick(in_idx), s=1, label=label)
        label = ins
        ax.scatter(xs=data[labels == label, 0], ys=data[labels == label, 1], zs=data[labels == label, 2],
                   c=color_pick(in_idx), marker=marker_pick(in_idx), s=2, label=label)

    plt.legend(bbox_to_anchor=(1.06, 0), fontsize='5', loc=3, borderaxespad=0)

    if fig_title is not None:
        plt.title(fig_title)


df = pd.read_csv('P300_600g_norm2.csv', index_col=0)

spectra = []
labels = []
for idx, row in df.iterrows():
    data = np.loadtxt(row['data_file'], delimiter='\t', dtype=float).T
    spectra.append(data[1])
    # label = row['label_name'].split('_')[0]
    # labels.append(label)
    labels.append(row['label_name'])
spectra = np.array(spectra)
labels = np.array(labels)

pca = ml.do_tSNE(spectra, components=2)

fig = plt.figure(figsize=(20, 10), dpi=300)
ax = fig.add_subplot(111)
class_list = ['ABA', 'CFR', 'ECO', 'EFA', 'KOX', 'KPN', 'PMA', 'SAU', 'SHO', 'SMA']
for idx, label in enumerate(class_list):
    ax.scatter(x=pca[labels == label, 0], y=pca[labels == label, 1], c=color_pick(idx), s=18, label=label)

plt.legend(bbox_to_anchor=(1.01, 0), fontsize='16', loc=3, borderaxespad=0)

plt.savefig('result/tSNE_2D_P300_600g.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
# plt.show()
