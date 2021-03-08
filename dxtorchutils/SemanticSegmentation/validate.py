import time
import cv2
import function
from dxtorchutils.utils.metrics import accuracy
from torch.utils.data import DataLoader
import numpy as np
from dxtorchutils.utils.info_logger import Logger
from dxtorchutils.utils.utils import state_logger
import torch
import matplotlib.pyplot as plt


class ValidateVessel:
    def __init__(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        model_paras_path: str = None,
        color_map=None
    ):
        self.model = model
        if model_paras_path is not None:
            self.model.load_state_dict(torch.load(model_paras_path))

        self.is_gpu = False
        self.dataloader = dataloader
        self.metrics = [accuracy]
        self.metric_names = ["accuracy"]
        self.logger = None
        self.is_tensorboard = False
        self.alter_raw_img_func = None
        self.alter_label_img_func = None
        self.alter_prediction_img_func = None

        if color_map is None:
            self.color_map = dataloader.dataset.color_map
        else:
            self.color_map = color_map

    def validate(self):
        state_logger("Model and Dataset Loaded, Start to Validate!")
        if self.logger is None and self.is_tensorboard:
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
                        pred = np.squeeze(prediction.cpu().data.numpy())
                        label = targets.cpu().data.numpy()
                        raw = data.cpu().data.numpy()
                    else:
                        pred = np.squeeze(prediction.data.numpy())
                        label = targets.data.numpy()
                        raw = data.data.numpy()

                    img_pred = (np.zeros((pred.shape[0], pred.shape[1], 3))).astype(np.uint8)
                    img_label = (np.zeros((label.shape[0], label.shape[1], 3))).astype(np.uint8)
                    img_raw = (np.reshape(raw, raw.shape[1:]).transpose(1, 2, 0) * 255).astype(np.uint8)

                    # prediction color map 回去
                    for pred_idx, color in self.color_map:
                        iS, jS = np.where(pred == pred_idx)[0: 2]
                        for i, j in zip(iS, jS):
                            img_pred[i][j] = color

                    # label color map 回去
                    for label_idx, color in self.color_map:
                        iS, jS = np.where(label == label_idx)[0: 2]
                        for i, j in zip(iS, jS):
                            img_label[i][j] = color

                    lw_rate = raw.shape[-1] / raw.shape[-2]
                    img_pred = cv2.resize(img_pred, (100, int(lw_rate * 100)))
                    img_label = cv2.resize(img_label, (100, int(lw_rate * 100)))
                    img_raw = cv2.resize(img_raw, (100, int(lw_rate * 100)))


                    if self.alter_raw_img_func is not None:
                        img_raw = self.alter_raw_img_func(img_raw)

                    if self.alter_label_img_func is not None:
                        img_label = self.alter_label_img_func(img_label)

                    if self.alter_prediction_img_func is not None:
                        img_pred = self.alter_prediction_img_func(img_raw)


                    if len(img_label.shape) == 2:
                        img_label = np.reshape(img_label, (img_label.shape[0], img_label.shape[1], 1))


                    reses = np.zeros(eval_res.shape)

                    for idx, metric in enumerate(self.metrics):
                        pre_res = eval_res[idx]
                        reses[idx] = metric(np.reshape(label, -1), np.reshape(pred, -1))
                        eval_res[idx] = (pre_res * iteration + reses[idx]) / (iteration + 1)


                    metrics_log = ""
                    metrics_tb_log = ""

                    for idx, (name, res) in enumerate(zip(self.metric_names, reses)):
                        if idx == len(self.metric_names) - 1:
                            metrics_tb_log += "{}:{:.3}".format(name, res)
                            metrics_log += "| {}: {:.3} |".format(name, res)
                        else:
                            metrics_tb_log += "{}:{:.3}/".format(name, res)
                            metrics_log += "| {}: {:.3} ".format(name, res)


                    if self.is_tensorboard:
                        fig = plt.figure()
                        plt.suptitle(metrics_tb_log)

                        plt.subplot(1, 3, 1)
                        plt.title("raw image")
                        plt.axis('off')
                        if img_raw.shape[-1] == 1:
                            plt.imshow(img_raw, cmap='gray')
                        else:
                            plt.imshow(img_raw)


                        plt.subplot(1, 3, 2)
                        plt.title("ground truth")
                        plt.axis('off')
                        plt.imshow(img_label)


                        plt.subplot(1, 3, 3)
                        plt.title("prediction")
                        plt.axis('off')
                        plt.imshow(img_pred)

                        self.logger.add_figure("Validate/{}".format(iteration), fig, iteration)

                    print(metrics_log)

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

    def enable_tensorboard(self, path=None):
        self.is_tensorboard = True
        if path is not None:
            self.logger = Logger(path)

    def multi_gpu(self, device_ids):
        self.is_gpu = True
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
