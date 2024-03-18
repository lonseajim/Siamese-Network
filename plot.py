import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class Color:
    nature_color = [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
        '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#8F62A4', '#B3B3B3', '#DE5A6A',
        '#4F81BD', '#C0504D', '#9BBB59', '#8064A2', '#4BACC6', '#F79646', '#C00000', '#FF0000', '#FFC000', '#FFFF00',
        '#92D050', '#00B050', '#00B0F0', '#0070C0', '#002060', '#7030A0', '#E7E6E6', '#A6A6A6', '#DD7E6B', '#FFA500',
        '#FFC000', '#FFFF00', '#92D050', '#00B050', '#00B0F0', '#0070C0', '#002060', '#7030A0', '#E7E6E6', '#A6A6A6',
    ]

    science_color = [
        '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#8F62A4', '#B3B3B3', '#DE5A6A',
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
        '#40E0D0', '#FF4500', '#E6E6FA', '#D8BFD8', '#FA8072', '#32CD32', '#808000', '#6B8E23', '#FAFAD2', '#ADFF2F',
        '#FF69B4', '#FFC0CB', '#FFE4B5', '#FFEBCD', '#FFFACD', '#FAF0E6', '#FAFAD2', '#FFF5EE', '#FFF8DC', '#FFFAF0',
        '#F8F8FF', '#F5F5F5', '#F0FFF0', '#FDF5E6', '#FFE4C4', '#FFDEAD', '#FFE4E1', '#F0FFFF', '#F5FFFA', '#FFFFF0',
    ]


def color_pick(idx=0):
    color_list = getattr(Color, 'nature_color')
    return color_list[idx % len(color_list)]


def marker_pick(idx=0):
    marker_list = ['o', '^', 's', 'd']
    return marker_list[idx % len(marker_list)]


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, pic_size=(8, 8)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    # print(cm)
    fig = plt.figure(figsize=pic_size)
    ax = fig.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_spectrum(x, y, fig_size=(8, 5)):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.plot(x, y, color=color_pick(0), linewidth=2)
    plt.xlabel('Wavenumber(cm$^-1$)', fontsize=10)
    plt.ylabel('Intensity', fontsize=10)


def plot_spectra(spec_dict: dict, label_dict: dict = None, x=None, single_height=3, width=18, fig_title=None,
                 show_legend=False):
    """
    Plot spectra
    :param spec_dict: 拉曼光谱字典 {'0':[[],[]], '1':[[],[]]}
    :param label_dict: 标签字典 {'0':'a', '1':'b'}
    :param x: []
    :param single_height: 单个光谱高度
    :param width: 图片宽度
    :param fig_title: 标题
    :param show_legend: 是否显示图例
    :return:
    """
    len_labels = len(spec_dict)
    # 计算均谱
    avg_spec_dict, label_dict2 = {}, {}
    for label_no, spec in spec_dict.items():
        avg_spec_dict[label_no] = sum(spec) / len(spec)
        label_dict2[label_no] = label_no
    # 标签字典
    if label_dict is None:
        label_dict = label_dict2
    # x轴
    if x is None:
        x = list(range(len(next(iter(avg_spec_dict.values())))))
    # 设置图像大小
    fig = plt.figure(figsize=(width, single_height * len_labels))

    for idx, (label_no, label_name) in enumerate(label_dict.items()):
        ax = fig.add_subplot(len_labels, 1, idx + 1)
        # 显示所有数据
        for data in spec_dict[label_no]:
            ax.plot(x, data, color='#dddddd')
        # 显示均线
        ax.plot(x, avg_spec_dict[label_no], color=color_pick(idx), linewidth=3, label=label_name)
        # 显示边框线
        if idx != len_labels - 1:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none')  # 隐藏 x 轴刻度线
            ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        if show_legend is False:
            ax.get_legend().remove()
    # 显示坐标轴等信息
    top = 0.98
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=18)
        top = 0.95
    fig.supxlabel('Wavenumber(cm$^-1$)', fontsize=16)
    fig.supylabel('Intensity', fontsize=16)
    fig.subplots_adjust(bottom=0.05, left=0.08, top=top)
    # plt.show()


def plot_spectra2(spec_dict: dict, label_dict: dict = None, x=None, offset=0.5, fig_size=(18, 15), fig_title=None,
                  show_legend=False, show_label=False):
    # len_labels = len(spec_dict)
    # 计算均谱
    avg_spec_dict, label_dict2 = {}, {}
    for label_no, spec in spec_dict.items():
        avg_spec_dict[label_no] = sum(spec) / len(spec)
        label_dict2[label_no] = label_no
    # 标签字典
    if label_dict is None:
        label_dict = label_dict2
    # x轴
    if x is None:
        x = list(range(len(next(iter(avg_spec_dict.values())))))
    # 设置图像大小
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.yaxis.set_major_locator(plt.NullLocator())

    for idx, (label_no, label_name) in enumerate(label_dict.items()):
        # 显示所有数据
        for data in spec_dict[label_no]:
            ax.plot(x, data - idx * offset, color='#dddddd')
        # 显示均线
        ax.plot(x, avg_spec_dict[label_no] - idx * offset, color=color_pick(idx), linewidth=2, label=label_name)

        ax.legend(bbox_to_anchor=(1.01, 0), fontsize='20', loc=3, borderaxespad=0)
        if show_legend is False:
            ax.get_legend().remove()
    # 显示坐标轴等信息
    if fig_title is not None:
        plt.suptitle(fig_title, fontsize=18)
    if show_label is True:
        plt.xlabel('Wavenumber(cm$^-1$)', fontsize=16)
        plt.ylabel('Intensity', fontsize=16)


def plot_2D_scatter(data: np.array, labels: np.array, fig_size=(14, 10), fig_dpi=100, fig_title=None):
    assert len(data) == len(labels)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    ax = fig.add_subplot(111)
    # 去重
    labels_distinct = list(set(labels))
    for idx, label in enumerate(labels_distinct):
        ax.scatter(x=data[labels == label, 0], y=data[labels == label, 1], c=color_pick(idx), s=15, label=label)
    # 显示图例
    plt.legend(bbox_to_anchor=(1.01, 0), fontsize='20', loc=3, borderaxespad=0)
    # 显示标题
    if fig_title is not None:
        plt.title(fig_title)
    # plt.show()


def plot_3D_scatter(data: np.array, labels: np.array, fig_size=(14, 10), fig_dpi=100, fig_title=None):
    assert len(data) == len(labels)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    ax = fig.add_subplot(111, projection='3d')
    # 去重
    labels_distinct = list(set(labels))
    labels_distinct.sort()
    for idx, label in enumerate(labels_distinct):
        ax.scatter(xs=data[labels == label, 0], ys=data[labels == label, 1], zs=data[labels == label, 2],
                   c=color_pick(idx), s=15, label=label)
    # 显示图例
    plt.legend(bbox_to_anchor=(1.01, 0), fontsize='20', loc=3, borderaxespad=0)
    # 显示标题
    if fig_title is not None:
        plt.title(fig_title)
    # plt.show()
