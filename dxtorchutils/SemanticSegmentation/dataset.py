import bisect
import os
import warnings

import function
from torch.utils import data
import cv2
import torch
import numpy as np
from torch.utils.data import IterableDataset

from dxtorchutils.utils.utils import state_logger


class DatasetConfig:
    def __init__(
            self,
            raw_dir_path: str,
            label_dir_path: str,
            raw_name_format: str = "{}.png",
            label_name_format: str = "{}.png",
            color_map=None,
    ):
        """
        :param raw_dir_path: 原图的文件夹名
        :param label_dir_path: 标签的文件夹名
        :param raw_name_format: 原图的文件命名格式，
        :param label_name_format: 标签的文件命名格式，{}中间包裹的是图片id，如：图片名字：raw_1001.png, raw_1002.png，则该处为 raw_{}.png
        :param color_map: 颜色的map，如果不填，自动读前10张的所有类型，填int类型，读前n张的所有类型，
                            e.g. [[0, [0, 0, 255], [1, [255, 0, 0]]]
        """
        super(DatasetConfig, self).__init__()


        assert os.path.exists(raw_dir_path), "Wrong raw path: {}".format(raw_dir_path)
        assert os.path.exists(label_dir_path), "Wrong raw path: {}".format(label_dir_path)

        self.raw_funcs = []
        self.label_funcs = []
        self.stop_at = None
        self.image_id = []
        if not raw_dir_path.endswith("/"):
            self.raw_dir_path = raw_dir_path + "/"
        else:
            self.raw_dir_path = raw_dir_path

        if not label_dir_path.endswith("/"):
            self.label_dir_path = label_dir_path + "/"
        else:
            self.label_dir_path = label_dir_path

        self.raw_suffix = raw_name_format.split(".")[-1]
        self.raw_head = raw_name_format.split("{")[0]
        self.raw_tail = raw_name_format.split("{")[-1].split("}")[-1].split(".")[0]

        self.label_suffix = label_name_format.split(".")[-1]
        self.label_head = label_name_format.split("{")[0]
        self.label_tail = label_name_format.split("{")[-1].split("}")[-1].split(".")[0]

        self.raw_suffix = "." + self.raw_suffix
        self.label_suffix = "." + self.label_suffix

        self.color_map = color_map

    def all_set_up(self):
        raw_names = os.listdir(self.raw_dir_path)
        label_names = os.listdir(self.label_dir_path)

        stop = 0
        for raw_name in raw_names:
            if raw_name.endswith(self.raw_suffix) and raw_name[0] != ".":
                if len(self.raw_head) < len(raw_name) - (len(self.raw_tail) + len(self.raw_suffix)):
                    condition1 = raw_name[:len(self.raw_head)] == self.raw_head
                    condition2 = raw_name[- (len(self.raw_tail) + len(self.raw_suffix)): - len(self.raw_suffix)] == self.raw_tail
                    condition3 = raw_name[- len(self.raw_suffix):] == self.raw_suffix
                    if condition1 and condition2 and condition3:
                        raw_id = raw_name[len(self.raw_head): - (len(self.raw_tail) + len(self.raw_suffix))]
                        label_name = self.label_head + raw_id + self.label_tail + self.label_suffix
                        if label_name in label_names:
                            raw = cv2.imread(self.raw_dir_path + self.raw_head + raw_id + self.raw_tail + self.raw_suffix)
                            label = cv2.imread(self.label_dir_path + self.label_head + raw_id + self.label_tail + self.label_suffix)
                            if raw is not None and label is not None:
                                self.image_id.append(raw_id)

                        if self.stop_at is not None:
                            stop += 1
                            if stop == self.stop_at:
                                break


        state_logger("Dataset Prepared! Num: {}".format(len(self.image_id)))


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


    def add_label_func(self, *label_funcs):
        """
        设置所需对原图改动的函数，输入cv2读入的原图，返回同样cv2可读的格式
        可只用lambda表达式
        e.g. dataset.add_label_func(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        :param label_funcs: 一个或多个function或lambda
        :return:
        """
        for label_func in label_funcs:
            self.label_funcs.append(label_func)


    def resize_label(self, dsize, dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST):
        """
        resize标签
        :param dsize:
        :param dst:
        :param fx:
        :param fy:
        :param interpolation:
        :return:
        """
        self.label_funcs.append(lambda img: cv2.resize(img, dsize, dst, fx, fy, interpolation))

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

        self.image_id = dataset_config.image_id
        self.raw_dir_path = dataset_config.raw_dir_path
        self.raw_head = dataset_config.raw_head
        self.raw_tail = dataset_config.raw_tail
        self.raw_suffix = dataset_config.raw_suffix
        self.label_dir_path = dataset_config.label_dir_path
        self.label_head = dataset_config.label_head
        self.label_tail = dataset_config.label_tail
        self.label_suffix = dataset_config.label_suffix
        self.raw_funcs = dataset_config.raw_funcs
        self.label_funcs = dataset_config.label_funcs

        if dataset_config.color_map is None:
            # 拿到color map
            label_paths = []
            for i in range(20):
                label_paths.append(
                    self.label_dir_path + self.label_head + self.image_id[i] + self.label_tail + self.label_suffix
                )
            self.color_map = get_color_map(label_paths)
        elif isinstance(dataset_config.color_map, int):
            # 拿到color map
            label_paths = []
            for i in range(dataset_config.color_map):
                label_paths.append(
                    self.label_dir_path + self.label_head + self.image_id[i] + self.label_tail + self.label_suffix
                )
            self.color_map = get_color_map(label_paths)
        else:
            self.color_map = dataset_config.color_map

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):
        image_id = self.image_id[index]

        if isinstance(image_id, list):
            data = []
            targets = []
            image_ids = image_id
            for image_id in image_ids:
                data, targets = self.get_data_target(image_id, data, targets)

        else:
            data, targets = self.get_data_target(image_id)

        data = torch.from_numpy(np.array(data)).type(torch.FloatTensor)
        targets = torch.from_numpy(np.array(targets)).type(torch.LongTensor)

        return data, targets

    def get_data_target(self, image_id, data=None, targets=None):
        raw_path = self.raw_dir_path + self.raw_head + image_id + self.raw_tail + self.raw_suffix
        label_path = self.label_dir_path + self.label_head + image_id + self.label_tail + self.label_suffix

        raw_image = cv2.imread(raw_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        label_image = cv2.imread(label_path)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)


        for raw_func in self.raw_funcs:
            raw_image = raw_func(raw_image)

        if len(np.array(list(raw_image.shape))) == 3:
            raw_image = raw_image.transpose(2, 0, 1)

        if len(np.array(list(raw_image.shape))) == 2:
            raw_image = np.reshape(raw_image, (1, raw_image.shape[0], raw_image.shape[1]))

        if raw_image.dtype == "uint8":
            raw_image = raw_image / 255

        for label_func in self.label_funcs:
            label_image = label_func(label_image)

        target_image = np.zeros(label_image.shape[:2])

        # 注意-1的操作，这样加快很多
        for idx, color in self.color_map:
            target_image = np.where((label_image == color).all(axis=-1), idx, target_image)

        if data is None:
            return raw_image, target_image
        else:
            data.append(raw_image)
            targets.append(target_image)

            return data, targets



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
        self.color_map = None
        if isinstance(dataset_configs[0], list):
            for dataset_config in dataset_configs[0]:
                dataset = SingleDataset(dataset_config)
                datasets.append(dataset)
                if self.color_map is None:
                    self.color_map = dataset.color_map
        else:
            for dataset_config in dataset_configs:
                dataset = SingleDataset(dataset_config)
                datasets.append(dataset)
                if self.color_map is None:
                    self.color_map = dataset.color_map

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


def get_color_map(*label_paths):
    """
    根据输入的路径，拿到color map
    :param label_paths: 标签路径
    :return:
    """
    if isinstance(label_paths[0], list):
        images = []
        for label_path in label_paths[0]:
            image = cv2.imread(label_path)
            assert image is not None, "{} doesn't exit image!".format(label_path)
            images.append(image)

        color_map = get_color_map_from_images(images)
    else:
        images = []
        for label_path in label_paths:
            image = cv2.imread(label_path)
            assert image is not None, "{} doesn't exit image!".format(label_path)
            images.append(image)

        color_map = get_color_map_from_images(images)
    return color_map


def get_color_map_from_images(*images):
    """
        根据输入的图片，拿到color map
        :param images: 标签路径
        :return:
    """
    image_tuples = []
    if isinstance(images[0], list):
        for image in images[0]:
            image = np.reshape(image, (-1, 3))
            for bgr in image:
                image_tuples.append(tuple(bgr))
    else:
        for image in images:
            image = np.reshape(image, (-1, 3))
            for bgr in image:
                image_tuples.append(tuple(bgr))

    colors = list(set(image_tuples))
    color_map = [[idx, color] for idx, color in enumerate(colors)]

    return color_map
