import os
import function
from torch.utils import data
import cv2
import torch
import numpy as np
from .Others import state_logger


class SemanticSegDataset(data.Dataset):
    def __init__(
            self,
            dataset_dir_path: str,
            dataset_type: str,
            raw_path: str = "raw",
            label_path: str = "label",
            raw_name_format: str = "{}",
            label_name_format: str = "{}",
            raw_func: function = None,
            label_func: function = None,
            color_map: [] = None,
            stop_at: int = None,
    ):
        super(SemanticSegDataset, self).__init__()
        assert dataset_type == "training" or "validation" or "testing", "Wrong dataset type!"

        if dataset_dir_path.endswith("/"):
            dataset_path = dataset_dir_path + dataset_type
        else:
            dataset_path = dataset_dir_path + "/" + dataset_type

        self.raw_dataset_path = dataset_path + "/" + raw_path + "/"
        self.label_dataset_path = dataset_path + "/" + label_path + "/"

        self.raw_suffix = raw_name_format.split(".")[-1]
        self.raw_head = raw_name_format.split("{")[0]
        self.raw_tail = raw_name_format.split("{")[-1].split("}")[-1].split(".")[0]

        self.label_suffix = label_name_format.split(".")[-1]
        self.label_head = label_name_format.split("{")[0]
        self.label_tail = label_name_format.split("{")[-1].split("}")[-1].split(".")[0]

        self.raw_suffix = "." + self.raw_suffix
        self.label_suffix = "." + self.label_suffix

        raw_names = os.listdir(self.raw_dataset_path)
        label_names = os.listdir(self.label_dataset_path)
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

        # 拿到color map
        label_path = self.label_dataset_path + self.label_head + self.image_id[0] + self.label_tail + self.label_suffix
        if color_map is None:
            self.color_map = get_color_map(label_path)
        else:
            self.color_map = color_map


        self.raw_func = raw_func
        self.label_func = label_func

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
        raw_path = self.raw_dataset_path + self.raw_head + image_id + self.raw_tail + self.raw_suffix
        label_path = self.label_dataset_path + self.label_head + image_id + self.label_tail + self.label_suffix

        raw_image = cv2.imread(raw_path)
        label_image = cv2.imread(label_path)

        if self.raw_func is None:
            if len(np.array(list(raw_image.shape))) == 3:
                raw_image = raw_image.transpose(2, 0, 1)
        else:
            raw_image = self.raw_func(raw_image)

        raw_image = raw_image / 255

        if self.label_func is not None:
            label_image = self.label_func(label_image)

        target_image = np.zeros(label_image.shape[:2])

        # 注意-1的操作，这样加快很多
        for idx, color in self.color_map:
            target_image = np.where((label_image == color).all(axis=-1), idx, target_image)

        if data is None:
            return raw_image, target_image
        else:
            data.append(raw_image)
            target.append(target_image)
            return data, target


def get_color_map(label_path):
    image = cv2.imread(label_path)
    image = np.reshape(image, (-1, 3))
    image_tuple = [tuple(bgr) for bgr in image]
    colors = list(set(image_tuple))
    color_map = []
    for idx, color in enumerate(colors):
        color_map.append([idx, color])

    return color_map
