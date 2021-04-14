import time
import cv2
import function
from dxtorchutils.utils.metrics import iou_macro
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
        self.metrics = [iou_macro]
        self.metric_names = ["miou"]
        self.metric_types = ["prediction"]
        self.logger = None
        self.is_tensorboard = False
        self.alter_raw_img_func = None
        self.alter_label_img_func = None
        self.alter_prediction_img_func = None
        self.tensorboard_display_image_width = 200


        if color_map is None:
            if hasattr(dataloader.dataset, "color_map"):
                self.color_map = dataloader.dataset.color_map
        else:
            self.color_map = color_map

        assert self.color_map is not None, "No color map provided!"

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

                print(log_metric)

                if self.is_tensorboard:
                    if self.is_gpu:
                        data = data.cpu()

                    # 转成对每张图进行操作
                    for pred, label, raw in zip(preds, labels, data.data.numpy()):
                        img_pred = (np.zeros((pred.shape[0], pred.shape[1], 3))).astype(np.uint8)
                        img_label = (np.zeros((label.shape[0], label.shape[1], 3))).astype(np.uint8)
                        img_raw = (raw.transpose(1, 2, 0) * 255).astype(np.uint8)

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

                        # 展示图的大小
                        lw_rate = raw.shape[-1] / raw.shape[-2]
                        width = self.tensorboard_display_image_width
                        img_pred = cv2.resize(img_pred, (width, int(lw_rate * width)))
                        img_label = cv2.resize(img_label, (width, int(lw_rate * width)))
                        img_raw = cv2.resize(img_raw, (width, int(lw_rate * width)))


                        # 对三个图的操作，输入输出都是cv2可读
                        if self.alter_raw_img_func is not None:
                            img_raw = self.alter_raw_img_func(img_raw)

                        if self.alter_label_img_func is not None:
                            img_label = self.alter_label_img_func(img_label)

                        if self.alter_prediction_img_func is not None:
                            img_pred = self.alter_prediction_img_func(img_raw)

                        if len(img_label.shape) == 2:
                            img_label = np.reshape(img_label, (img_label.shape[0], img_label.shape[1], 1))

                        # 灰度图接受二维矩阵
                        raw = np.squeeze(raw)

                        # iou结果
                        iou_res = iou_macro(targets.data.numpy(), predictions.data.numpy())

                        # 画表格
                        fig = plt.figure()

                        # 大标题
                        plt.suptitle("IoU: {:.4}".format(iou_res))

                        # 横着分三栏
                        plt.subplot(1, 3, 1)
                        # 第一个原图
                        plt.title("raw image")
                        # 取消坐标轴
                        plt.axis('off')
                        # 灰度图
                        if len(raw.shape) == 2:
                            plt.imshow(img_raw, cmap='gray')
                        else:
                            plt.imshow(img_raw)

                        # 标签
                        plt.subplot(1, 3, 2)
                        plt.title("ground truth")
                        plt.axis('off')
                        plt.imshow(img_label)

                        # 预测
                        plt.subplot(1, 3, 3)
                        plt.title("prediction")
                        plt.axis('off')
                        plt.imshow(img_pred)

                        self.logger.add_figure("Validation/{}".format(iteration), fig, iteration)

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


    def set_tensorboard_display_image_width(self, width):
        self.tensorboard_display_image_width = width
