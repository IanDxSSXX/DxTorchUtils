import time
from torch.utils.data import DataLoader
from dxtorchutils.utils.metrics import accuracy
import numpy as np
from dxtorchutils.utils.utils import state_logger
from dxtorchutils.utils.info_logger import Logger
import random
import torch


class TrainVessel:
    def __init__(
            self,
            dataloader: DataLoader,
            model: torch.nn.Module,
            model_paras_path: str = None,
            epochs: int = 20,
            opt: torch.optim = None,
            criteria: torch.nn.Module = None,
            eval_num: int = 100,
    ):
        """
           训练器
           :param dataloader: 传入的需要训练的dataloader
           :param model: 需要训练的模型
           :param model_paras_path: 训练好的参数的地址，默认为空即重新开始训练
           :param opt: 优化器，默认用SGD
           :param criteria: 损失函数，默认用交叉熵
           :param epochs: 训练循环次数，默认20
           :param eval_num: 每五个epoch，随机取eval_num个来测试已训练的模型，这是选取数量，默认10个
           :return:
        """

        self.model = model
        self.is_gpu = False
        self.is_multi_gpu = False
        self.is_tensorboard = False
        self.dataloader = dataloader
        self.epochs = epochs
        self.eval_num = eval_num
        self.eval_metric_func = accuracy
        self.eval_metric_name = "accuracy"
        self.logger = None

        self.model_save_path = None
        self.time_start = None
        self.iteration = None
        self.loss_all = None
        self.eval_res_all = None
        self.log_every_epoch = 1

        if model_paras_path is not None:
            self.model.load_state_dict(torch.load(model_paras_path))

        if opt is None:
            self.opt = torch.optim.SGD(model.parameters(), 5e-4, 0.9)

        if criteria is None:
            self.criteria = torch.nn.CrossEntropyLoss()

    def train(self):
        state_logger("Model and Dataset Loaded, Start to Train!")
        if self.logger is None and self.is_tensorboard is True:
            self.logger = Logger("logger/{}-{}".format(self.model.__class__.__name__.lower(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        self.time_start = time.time()

        if self.is_gpu:
            self.model = self.model.cuda()

        self.model.train()

        self.iteration = 0

        for epoch in range(self.epochs):
            for data, targets in self.dataloader:
                self.train_mini_batch(data, targets)

            if (epoch + 1) % self.log_every_epoch == 0 or epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    self.loss_all = []
                    self.eval_res_all = []

                    rand = random.randint(0, len(self.dataloader.dataset) - self.eval_num - 1)
                    for index in range(rand, rand + self.eval_num):
                        data, target = self.dataloader.dataset.__getitem__(index)

                        if self.is_gpu:
                            data = data.cuda()
                            target = target.cuda()

                        data = torch.unsqueeze(data, 0)
                        target = torch.unsqueeze(target, 0)

                        self.eval_through_training(data, target)

                    eval_res = np.mean(np.array(self.eval_res_all))
                    loss = np.mean(np.array(self.loss_all))

                    print("Epoch: {:04}/{:04} | Loss: {:.5} | {}: {:.5}"
                          .format(epoch + 1, self.epochs, loss, self.eval_metric_name, eval_res))

                    if self.is_tensorboard:
                        self.logger.log_metric(eval_res, self.eval_metric_name, loss, epoch)

                    if self.model_save_path is not None:
                        if self.is_multi_gpu:
                            torch.save(self.model.module.state_dict(), self.model_save_path)
                        else:
                            torch.save(self.model.state_dict(), self.model_save_path)
                    else:
                        if self.is_multi_gpu:
                            torch.save(self.model.module.state_dict(), self.model.__class__.__name__.lower() + ".pth")
                        else:
                            torch.save(self.model.state_dict(), self.model.__class__.__name__.lower() + ".pth")

                self.model.train()

            torch.cuda.empty_cache()

        if self.is_tensorboard:
            input_data, _ = next(iter(self.dataloader))
            if self.is_gpu:
                input_data = input_data.cuda()
            self.logger.add_graph(self.model, input_data)
            self.logger.close()

        state_logger("Training Completed!")

    def train_mini_batch(self, data, targets):
        self.opt.zero_grad()

        if self.is_gpu:
            data = data.cuda()
            targets = targets.cuda()

        output = self.model(data)
        if isinstance(output, tuple):
            loss = self.criteria(output[0], targets)
            for i in range(1, len(output)):
                loss += 0.3 * self.criteria(output[i], targets)
        elif isinstance(output, list):
            loss = self.criteria(output[0], targets)
            for i in range(1, len(output)):
                loss += self.criteria(output[i], targets)
        else:
            loss = self.criteria(output, targets)
        loss.backward()

        self.opt.step()

        time_end = time.time()
        if self.is_tensorboard:
            self.logger.log_training(loss, self.opt.defaults["lr"], time_end - self.time_start, self.iteration)
        self.iteration += 1

    def eval_through_training(self, data, targets):
        output = self.model(data)
        loss = self.criteria(output, targets)

        if self.is_gpu:
            prediction = np.reshape(torch.max(output, 1)[1].cpu().data.numpy(), -1)
            targets = np.reshape(targets.cpu().data.numpy(), -1)
            loss_num = loss.cpu()
        else:
            prediction = np.reshape(torch.max(output, 1)[1].data.numpy(), -1)
            targets = np.reshape(targets.data.numpy(), -1)

            loss_num = loss

        self.eval_res_all.append(self.eval_metric_func(targets, prediction))
        self.loss_all.append(loss_num)

    def replace_eval_metric(self, metric_name, metric_func):
        self.eval_metric_name = metric_name
        self.eval_metric_func = metric_func

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

    def save_model_to(self, path):
        self.model_save_path = path

    def multi_gpu(self, device_ids):
        self.is_multi_gpu = True
        self.is_gpu = True
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


    def tensorboard_log_model(self):
        if self.logger is None:
            self.logger = Logger("logger/{}-{}".format(self.model.__class__.__name__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if self.is_gpu:
            self.model = self.model.cuda()

        input_data, _ = next(iter(self.dataloader))

        if self.is_gpu:
            input_data = input_data.cuda()

        self.logger.add_graph(self.model, input_data)
        self.logger.close()
