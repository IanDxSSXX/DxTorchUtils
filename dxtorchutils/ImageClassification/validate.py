import time

import cv2
import function
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import numpy as np
from ..utils.info_logger import Logger
from ..utils.utils import state_logger
import torch


class ValidateVessel:
    def __init__(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        model_paras_path: str = None,
    ):
        self.model = model
        if model_paras_path is not None:
            self.model.load_state_dict(torch.load(model_paras_path))

        self.is_gpu = False
        self.dataloader = dataloader
        self.metrics = [accuracy_score]
        self.metric_names = ["accuracy"]
        self.logger = None
        self.is_tensorboard = False

    def validate(self):
        state_logger("Model and Dataset Loaded, Start to Validate!")
        self.logger = Logger("logger/{}-{}".format(self.model.__class__.__name__.lower(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if self.is_gpu:
            self.model.cuda()

        self.model.eval()

        iteration = 1
        eval_res = np.zeros(len(self.metrics))
        with torch.no_grad():
            for val_data, val_target in self.dataloader:
                for data, targets in zip(val_data, val_target):
                    if self.is_gpu:
                        data = data.cuda()
                        targets = targets.cuda()

                    data = torch.unsqueeze(data, 0)
                    output = self.model(data)

                    prediction = torch.max(output, 1)[1].type(torch.LongTensor)
                    targets = targets.type(torch.LongTensor)

                    if self.is_gpu:
                        pred = prediction.cpu().data.numpy()
                        label = targets.cpu().data.numpy()
                        raw = data.cpu().data.numpy()
                    else:
                        pred = prediction.data.numpy()
                        label = targets.data.numpy()
                        raw = data.data.numpy()

                    raw = np.reshape(raw, raw.shape[1:]).transpose(1, 2, 0)
                    lw_rate = raw.shape[0] / raw.shape[1]
                    raw = (cv2.resize(raw, (100, int(lw_rate * 100))) * 255).astype(np.uint8)
                    if len(raw.shape) == 2:
                        raw = np.reshape(raw, (raw.shape[0], raw.shape[1], 1))

                    if self.is_tensorboard:
                        if label == pred[0]:
                            self.logger.add_image("True/Label:{} -- Pred:{}".format(label, pred[0]), raw, iteration, dataformats="HWC")
                        else:
                            self.logger.add_image("False/Label:{} -- Pred:{}".format(label, pred[0]), raw, iteration, dataformats="HWC")


                    for idx, metric in enumerate(self.metrics):
                        pre_res = eval_res[idx]
                        next_res = metric(np.reshape(label, -1), np.reshape(pred, -1))
                        eval_res[idx] = (pre_res * iteration + next_res) / (iteration + 1)

                    if iteration % 10 == 0:
                        for name, res in zip(self.metric_names, eval_res):
                            print("| {}: {:.3} ".format(name, res), end="")
                        print("|\n")

                    iteration += 1

        state_logger("Validate Completed!")

    def add_metric(self, metric_name, metric_func: function):
        self.metrics.append(metric_func)
        self.metric_names.append(metric_name)

    def gpu(self):
        self.is_gpu = True

    def cpu(self):
        self.is_gpu = False

    def load_model_para(self, model_paras_path: str):
        self.model.load_state_dict(torch.load(model_paras_path))

    def set_tensorboard_dir(self, path):
        self.is_tensorboard = True
        self.logger = Logger(path)

    def disable_tensorboard(self):
        self.is_tensorboard = False

    def enable_tensorboard(self):
        self.is_tensorboard = True

    def multi_gpu(self, device_ids):
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
