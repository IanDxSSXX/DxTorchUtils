import bisect
import os
import warnings

from torch.utils import data
import cv2
import torch
import numpy as np
from torch.utils.data import IterableDataset, Subset

from dxtorchutils.utils.utils import state_logger


class DatasetConfig:
    def __init__(
            self,
            raw_dir_path: str,
            label=None
    ):
        """
        :param raw_dir_path: 原图的文件夹名
        """
        super(DatasetConfig, self).__init__()

        assert os.path.exists(raw_dir_path), "Wrong raw path: {}".format(raw_dir_path)

        self.raw_funcs = []
        self.lc = []
        self.data = []
        self.targets = []
        self.stop_at = None
        self.label = label

        if not raw_dir_path.endswith("/"):
            self.raw_dir_path = raw_dir_path + "/"
        else:
            self.raw_dir_path = raw_dir_path


    def all_set_up(self):
        # 拿到所有的 raw name
        raw_names = os.listdir(self.raw_dir_path)

        stop = 0
        for raw_name in raw_names:
            if not raw_name.startswith("."):
                raw_dir = self.raw_dir_path + raw_name
                self.data.append(raw_dir)
                if self.label is None:
                    # 根据condition得到label
                    for lb, condition in self.lc:
                        if condition(raw_name):
                            self.targets.append(lb)

                else:
                    self.targets.append(self.label)

                if self.stop_at is not None:
                    stop += 1
                    if stop == self.stop_at:
                        break

        state_logger("Dataset Prepared! Num: {}".format(len(self.data)))

    def resize_raw(self, dsize, dst=None, fx=None, fy=None, interpolation=None):
        """
        resize原图
        :param dsize:
        :param dst:
        :param fx:
        :param fy:
        :param interpolation:
        :return:
        """
        self.raw_funcs.append(lambda img: cv2.resize(img, dsize, dst, fx, fy, interpolation))


    def cvt_color_raw(self, code=cv2.COLOR_RGB2GRAY):
        """
        原图改变颜色顺序，默认BGR转灰度图
        :param code:
        :return:
        """
        self.raw_funcs.append(lambda img: cv2.cvtColor(img, code))


    def add_raw_func(self, *raw_funcs):
        """
        设置所需对原图改动的函数，输入cv2读入的原图，返回同样cv2可读的格式
        可只用lambda表达式
        e.g. dataset.add_raw_func(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        :param raw_funcs: 一个或多个function或lambda
        :return:
        """
        for raw_func in raw_funcs:
            self.raw_funcs.append(raw_func)


    def set_label_condition_by_raw_name(self, *label_condition):
        """
        设置对应的label与名字的关系
        e.g. dataset.set_label_condition_by_raw_name
                (
                    [0, lambda raw_name: raw_name[:3] == "Id0"],
                    [1, lambda raw_name: raw_name[:3] == "Id1"]
                )
        :param label_condition: 列表中第二列一定要是function
        :return:
        """
        for lc in label_condition:
            self.lc.append(lc)


    def stop_at_idx(self, stop_at):
        """
        只读前stop_at张图，多用于测试
        :param stop_at:
        :return:
        """
        self.stop_at = stop_at


class SingleDataset(data.Dataset):
    def __init__(self, dataset_config: DatasetConfig):
        super(SingleDataset, self).__init__()
        dataset_config.all_set_up()

        self.data = dataset_config.data
        self.targets = dataset_config.targets
        self.raw_funcs = dataset_config.raw_funcs


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        raw_path = self.data[index]
        label = self.targets[index]

        if isinstance(raw_path, list):
            data = []
            raw_paths = raw_path
            for raw_path in raw_paths:
                data = self.get_data_target(raw_path, data)
        else:
            data = self.get_data_target(raw_path)

        data = torch.from_numpy(np.array(data)).type(torch.FloatTensor)
        targets = torch.from_numpy(np.array(label)).type(torch.LongTensor)

        return data, targets


    def get_data_target(self, raw_path, data=None):
        raw_image = cv2.imread(raw_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        for raw_func in self.raw_funcs:
            raw_image = raw_func(raw_image)

        if len(np.array(list(raw_image.shape))) == 3:
            raw_image = raw_image.transpose(2, 0, 1)

        if len(np.array(list(raw_image.shape))) == 2:
            raw_image = np.reshape(raw_image, (1, raw_image.shape[0], raw_image.shape[1]))

        if raw_image.dtype == "uint8":
            raw_image = raw_image / 255

        if data is None:
            return raw_image
        else:
            data.append(raw_image)

            return data



class Dataset(data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, *dataset_configs):
        super(Dataset, self).__init__()
        datasets = []
        if isinstance(dataset_configs[0], list):
            for dataset_config in dataset_configs[0]:
                datasets.append(SingleDataset(dataset_config))
        else:
            for dataset_config in dataset_configs:
                datasets.append(SingleDataset(dataset_config))

        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
