import os
import re
import function
from torch.utils import data
import cv2
import torch
import numpy as np
from ..utils.utils import state_logger


class Dataset(data.Dataset):
    def __init__(
            self,
            raw_dir_path: str,
            label_dir_path: str,
            raw_name_format: str,
            label_name_format: str,
            raw_func: function = None,
            stop_at: int = None,
    ):
        """
        :param raw_dir_path: 原图的文件夹名
        :param label_dir_path: 标签的文件夹名
        :param raw_name_format: 原图的文件命名格式，
        :param label_name_format: 标签的文件命名格式，{}中间包裹的是图片id，如：图片名字：raw_1001.png, raw_1002.png，则该处为 raw_{}.png
        :param raw_func: 所需对原图改动的函数，输入cv2读入的原图，返回同样cv2可读的格式
        :param stop_at: 只读前stop_at张图，多用于测试，默认不管
        """
        super(Dataset, self).__init__()

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

        raw_names = os.listdir(self.raw_dir_path)
        label_names = os.listdir(self.label_dir_path)
        self.image_id = []

        stop = 0
        for raw_name in raw_names:
            if raw_name.endswith(self.raw_suffix) and raw_name[0] != ".":
                raw_id = raw_name.strip(self.raw_head).strip(self.raw_tail).strip(self.raw_suffix)
                label_name = self.label_head + raw_id + self.label_tail + self.label_suffix
                if label_name in label_names:
                    self.image_id.append(raw_id)

                if stop_at is not None:
                    if stop == stop_at:
                        break
                    stop += 1

        self.raw_func = raw_func

        state_logger("Dataset Prepared!")

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):
        image_id = self.image_id[index]

        if isinstance(image_id, list):
            data = []
            target = []
            image_ids = image_id
            for image_id in image_ids:
                data, target = self.get_data_target(image_id, data, target)

        else:
            data, target = self.get_data_target(image_id)

        data = torch.from_numpy(np.array(data)).type(torch.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.LongTensor)

        return data, target

    def get_data_target(self, image_id, data=None, target=None):
        raw_path = self.raw_dir_path + self.raw_head + image_id + self.raw_tail + self.raw_suffix
        label_path = self.label_dir_path + self.label_head + image_id + self.label_tail + self.label_suffix

        raw_image = cv2.imread(raw_path)
        with open(label_path, "r") as f:
            label_str = (f.read())
            label = int(re.sub("\D", "", label_str))

        if self.raw_func is not None:
            raw_image = self.raw_func(raw_image)

        if len(np.array(list(raw_image.shape))) == 3:
            raw_image = raw_image.transpose(2, 0, 1)

        if raw_image.dtype == "uint8":
            raw_image = raw_image / 255


        if data is None:
            return raw_image, label
        else:
            data.append(raw_image)
            target.append(label)
            return data, target
