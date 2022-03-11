import time
import cv2
import function
from torchtoy.utils.metrics import accuracy
from torch.utils.data import DataLoader
import numpy as np
from torchtoy.utils.info_logger import Logger
from torchtoy.utils.utils import state_logger
import torch
import matplotlib.pyplot as plt


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
        self.metrics = [accuracy]
        self.metric_names = ["accuracy"]
        self.metric_types = ["prediction"]
        self.logger = None
        self.is_tensorboard = False
        self.alter_raw_img_func = None

    def validate(self):
        state_logger("Model and Dataset Loaded, Start to Validate!")
        if self.logger is None and self.is_tensorboard:
            self.logger = Logger("logger/{}-{}".format(self.model.__class__.__name__.lower(),
                                                       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if self.is_gpu:
            self.model.cuda()

        self.model.eval()

        eval_res = np.zeros(len(self.metrics))
        eval_res_mean = np.zeros(len(self.metrics))
        count = 0
        iteration = 0
        with torch.no_grad():
            for data, targets in self.dataloader:
                if self.is_gpu:
                    data = data.cuda()
                    targets = targets.cuda()

                output = self.model(data)

                predictions = torch.max(output, 1)[1].type(torch.LongTensor)
                targets = targets.type(torch.LongTensor)

                # 开始算准确度，转cpu
                if self.is_gpu:
                    predictions = predictions.cpu()
                    targets = targets.cpu()

                preds = predictions.data.numpy()
                labels = targets.data.numpy()

                # 遍历所有的评价指标
                for idx, metric in enumerate(self.metrics):
                    # 上一次的结果
                    pre_res = eval_res_mean[idx]
                    # 这一次的结果
                    if self.metric_types[idx] == "prediction":
                        eval_res[idx] = metric(np.reshape(labels, -1), np.reshape(preds, -1))
                    else:
                        temp_preds = np.reshape(preds, -1)
                        temp_labels = np.reshape(labels, (len(temp_preds), -1))
                        eval_res[idx] = metric(temp_labels, temp_preds)

                    # 这一次平均值的结果
                    eval_res_mean[idx] = (pre_res * count + eval_res[idx]) / (count + 1)

                count += 1

                # 得到log信息
                log_metric = "| Batch: {} |".format(count)
                for name, res_mean, res in zip(self.metric_names, eval_res_mean, eval_res):
                    log_metric += "| {}: {:.4}(Mean) {:.4}(ThisBatch) ".format(name, res_mean, res)

                log_metric += "|"
                print(log_metric)

                if self.is_tensorboard:
                    if self.is_gpu:
                        data = data.cpu()

                    raws = data.data.numpy()

                    for raw, pred, label in zip(raws, preds, labels):
                        # 由 [C, H, W] 转为 [H, W, C]
                        raw = raw.transpose(1, 2, 0)
                        # resize成宽为150的图便于展示
                        lw_rate = raw.shape[0] / raw.shape[1]
                        raw = (cv2.resize(raw, (150, int(lw_rate * 150))) * 255).astype(np.uint8)

                        # 如果有一些对图像的其他操作，在这里
                        if self.alter_raw_img_func is not None:
                            self.alter_raw_img_func(raw)

                        # 灰度图接受二维矩阵
                        raw = np.squeeze(raw)

                        # 用plt画图，cv2有问题
                        fig = plt.figure()
                        # 取消坐标
                        plt.axis('off')

                        # 区分灰度图和彩色
                        if len(raw.shape) == 2:
                            plt.imshow(raw, cmap="gray")
                            plt.title("Label:{} -- Pred:{}".format(label, pred))
                        else:
                            plt.imshow(raw)
                            plt.title("Label:{} -- Pred:{}".format(label, pred))

                        # 区分正误
                        if label == pred:
                            self.logger.add_figure("ResultImage-True/{}".format(iteration), fig, iteration)
                        else:
                            self.logger.add_figure("ResultImage-False/{}".format(iteration), fig, iteration)

                        iteration += 1

        res_log = "Validation Completed With Result:"
        for name, res_mean in zip(self.metric_names, eval_res_mean):
            res_log += " {}: {:.4} ".format(name, res_mean)

        state_logger(res_log)


    def add_metric(self, metric_name, metric_func: function, metric_type="prediction"):
        assert metric_type in ["prediction", "output"]
        self.metrics.append(metric_func)
        self.metric_names.append(metric_name)
        self.metric_types.append(metric_type)



    def gpu(self):
        self.is_gpu = True


    def cpu(self):
        self.is_gpu = False


    def load_model_paras(self, model_paras_path: str):
        self.model.load_state_dict(torch.load(model_paras_path))


    def enable_tensorboard(self, path=None):
        self.is_tensorboard = True
        if path is not None:
            self.logger = Logger(path)


    def multi_gpu(self, device_ids):
        self.is_gpu = True
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
