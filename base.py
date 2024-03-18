"""
拉曼数据读取
"""

import os
import warnings

import numpy as np
import pandas as pd
import process_fun as process_fun

DATA_FILE_SUFFIX = ['.txt', '.npy']


class Spectrum:
    """
    读取单个拉曼光谱文件，文件格式为['.txt', '.npy']
    """

    def __init__(self, data_file: str, cut_range: tuple = None, process_flag: bool = False):
        # splitext(file)[0]获取文件名，splitext(file)[-1]获取带‘.’的后缀名
        data_file_suffix = os.path.splitext(data_file)[-1]
        assert data_file_suffix in DATA_FILE_SUFFIX
        # file
        self.data_file = data_file.replace('\\', '/')
        self.cut_range = cut_range
        # 获取文件名，带后缀
        self.data_file_name = os.path.split(self.data_file)[-1]
        # data
        self.data = None
        self.data_shape = None
        self.data_x = None
        self.data_y = None
        # axis
        self.x_axis = 'Wavenumber(cm-1)'
        self.y_axis = 'Intensity'

        if data_file_suffix == '.npy':
            self.readNpy()
        elif data_file_suffix == '.txt':
            self.readTxt()

        self.init()
        if process_flag:
            self.process()

    @property
    def file_name(self):
        return self.data_file_name

    @property
    def file_path(self):
        return self.data_file

    @property
    def raw(self):
        """
        Read raw data
        :return: self.data
        """
        return self.data

    @property
    def shape(self):
        """
        Read shape
        :return: self.data_shape
        """
        return self.data_shape

    @property
    def x(self):
        """
        Read x-axis
        :return: self.data_x
        """
        return self.data_x

    @property
    def y(self):
        """
        Read y-axis
        :return: self.data_y
        """
        return self.data_y

    @property
    def axis(self):
        return self.x_axis, self.y_axis

    @property
    def new_data(self):
        """
        Read new data
        :return: [self.data_x, self.data_y]
        """
        return np.array([self.data_x, self.data_y])

    def readNpy(self):
        """
        read data from a .npy file
        data format:
            shape: (n, ) without x-axis:
                [y]
            shape: (2, n):
                [x, y]
            shape: (m, n) where m > 2:
                [x, y(1), y(2),... y(m-1)]
        """
        self.data = np.load(self.data_file)

    def readTxt(self):
        """
        read data from a .txt file
        convert format like data from a .npy file
        """
        with open(self.data_file, 'r') as txt:
            lines = txt.readlines()
        ignore_rows = 0
        for idx, line in enumerate(lines):
            if line.startswith('#'):
                continue
            elif not line[0].isdigit():
                # 将首列数上方的列名作为X坐标轴名
                self.x_axis = line.split('\t')[0]
                continue
            items = line.split('\t')
            try:
                float(items[0])
                ignore_rows = idx
                break
            except ValueError:
                continue
        self.data = np.loadtxt(self.data_file, delimiter='\t', skiprows=ignore_rows).T

    def init(self):
        self.data_shape = self.data.shape
        if len(self.data_shape) == 1:
            self.data_x = None
            if self.cut_range is not None:
                warnings.warn('Data has only 1 dimension, cut range is ignored')
            self.data_y = self.data
        elif len(self.data_shape) == 2:
            self.data_x = self.data[0]
            cut_start, cut_end = 0, len(self.data_x)
            if self.cut_range is not None:
                cut_start = np.where(self.data_x == self.cut_range[0])[0][0]
                cut_end = np.where(self.data_x == self.cut_range[1])[0][0] + 1
                self.data_x = self.data_x[cut_start:cut_end]
            if self.data_shape[0] == 2:
                self.data_y = self.data[1][cut_start:cut_end]
            else:  # self.data_shape[0] > 2
                ys = self.data[1:, cut_start:cut_end]
                self.data_y = sum(ys) / len(ys)
        else:  # len(self.data_shape) > 2
            raise ValueError('No support for data shape')

    def process(self):
        """
        step 1：去尖峰（宇宙射线）
        step 2：去基线
        step 3：归一化
        :return: self.data_y
        """
        data = process_fun.despiking(self.data_y)
        data = process_fun.airPLS(data)
        self.data_y = process_fun.max_min_normalize(data)

    def save(self, file_name, suffix=None):
        if suffix is None:
            if file_name.endswith('.txt'):
                suffix = 'txt'
            else:
                suffix = 'npy'

        if 'npy' in suffix:
            np.save(file_name, self.new_data)
        elif 'txt' in suffix:
            if not file_name.endswith('.txt'):
                file_name = file_name + '.txt'
            np.savetxt(file_name, self.new_data.T, delimiter='\t')
        else:
            raise ValueError('Not a valid file suffix')

    def __str__(self):
        return ('File name: {}\nFile path: {}\nData shape: {}\nCut range: {}'
                .format(self.data_file_name, self.data_file, self.data_shape, self.cut_range))

    def __repr__(self):
        return ('File name: {}\nFile path: {}\nData shape: {}\nCut range: {}'
                .format(self.data_file_name, self.data_file, self.data_shape, self.cut_range))


class Spectra:
    """
    读取文件夹下的所有光谱文件，item返回Spectrum
    """

    def __init__(self, data_folder_path, cut_range: tuple = None, process_flag: bool = False):
        self.data_folder_path = data_folder_path
        self.cut_range = cut_range
        self.process_flag = process_flag
        self.data_folder_name = os.path.split(self.data_folder_path)[-1]
        # 获取以txt和npy后缀的文件列表
        file_list = os.listdir(self.data_folder_path)
        self.data_file_list = [os.path.join(self.data_folder_path, x).replace('\\', '/') for x in file_list if
                               x.endswith('.txt') or x.endswith('.npy')]
        self.data_x = Spectrum(self.data_file_list[0], cut_range=self.cut_range, process_flag=self.process_flag).x

    @property
    def file_list(self):
        return self.data_file_list

    def folder_name(self):
        return self.data_folder_name

    @property
    def x(self):
        return self.data_x

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, idx):
        return Spectrum(self.data_file_list[idx], cut_range=self.cut_range, process_flag=self.process_flag)

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame({'data_file': self.data_file_list})

    def get_raw_list(self) -> list:
        raw_list = []
        for file in self.data_file_list:
            spec = Spectrum(file, cut_range=self.cut_range, process_flag=self.process_flag)
            raw_list.append(spec.y)
        return raw_list


class SpectraFolder:
    """
    读取“文件夹/子文件夹/”下的所有光谱文件，并以子文件夹作为分类名
    """

    def __init__(self, folder_path, cut_range: tuple = None, process_flag: bool = False):
        self.folder_path = folder_path
        self.cut_range = cut_range
        self.process_flag = process_flag
        # 根据文件夹下的子文件夹作为分类名
        folder_list = os.listdir(self.folder_path)
        self.data_folder_list = [x for x in folder_list if '.' not in x]

        self.data_file_list = []
        self.label_list = []
        self.label_dict = {}
        for idx, data_folder in enumerate(self.data_folder_list):
            self.label_dict[str(idx)] = data_folder
            spectra = Spectra(os.path.join(self.folder_path, data_folder), cut_range=self.cut_range,
                              process_flag=self.process_flag)
            self.data_file_list = np.concatenate((self.data_file_list, spectra.file_list), axis=0)
            self.label_list = np.concatenate((self.label_list, np.ones(len(spectra)) * idx), axis=0)

        self.data_x = Spectra(os.path.join(self.folder_path, self.data_folder_list[0]), cut_range=self.cut_range,
                              process_flag=self.process_flag).x

    @property
    def files(self):
        return self.data_file_list

    @property
    def labels(self):
        return self.label_list

    @property
    def labels_info(self) -> dict:
        return self.label_dict

    @property
    def x(self):
        return self.data_x

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, idx):
        return (Spectrum(self.data_file_list[idx], cut_range=self.cut_range, process_flag=self.process_flag),
                self.label_list[idx])

    def get_df(self) -> pd.DataFrame:
        label_name_list = [self.label_dict[str(int(x))] for x in self.label_list]
        return pd.DataFrame({'data_file': self.data_file_list, 'label': self.label_list, 'label_name': label_name_list})

    def get_raw_dict(self):
        raw_dict = {}
        for label_no, label_name in self.label_dict.items():
            spectra = Spectra(os.path.join(self.folder_path, label_name).replace('\\', '/'), cut_range=self.cut_range,
                              process_flag=self.process_flag)
            raw_dict[label_no] = spectra.get_raw_list()
        return raw_dict

