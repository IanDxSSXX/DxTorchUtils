import function
from ..models.UNet import UNet
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from .Others import state_logger
import random


def train(
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        opt: torch.optim = None,
        criteria: torch.nn.Module = None,
        epochs: int = 100,
        gpu: bool = False,
        model_path: str = "model.pth",
        eval_func: function = None,
        eval_num: int = 10,
        test_working: bool = False,
):
    if model is None:
        if model_path is None:
            model = UNet()
        else:
            model = torch.load(model_path)

    if opt is None:
        opt = torch.optim.Adam(model.parameters())

    if criteria is None:
        criteria = torch.nn.CrossEntropyLoss()

    if gpu:
        model = model.cuda()

    state_logger("Model and Dataset Loaded, Start to Train!")

    for epoch in range(epochs):
        for idx, (data, target) in enumerate(dataloader):
            if gpu:
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss = criteria(output, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if test_working:
                break

        if test_working:
            break

        if epoch % 5 == 0:
            with torch.no_grad():
                rand = random.randint(0, len(dataloader.dataset) - eval_num - 1)
                test_data = dataloader.dataset.__getitem__(slice(rand, rand + eval_num))[0]
                test_target = dataloader.dataset.__getitem__(slice(rand, rand + eval_num))[1]

                if gpu:
                    test_data = test_data.cuda()
                    test_target = test_target.cuda()

                test_output = model(test_data)

                loss = criteria(test_output, test_target)

                if gpu:
                    test_prediction = np.reshape(torch.max(test_output, 1)[1].cpu().data.numpy(), -1)
                    test_target = np.reshape(test_target.cpu().data.numpy(), -1)

                    loss_num = loss.cpu()

                else:
                    test_prediction = np.reshape(torch.max(test_output, 1)[1].data.numpy(), -1)
                    test_target = np.reshape(test_target.data.numpy(), -1)

                    loss_num = loss

                if eval_func is None:
                    eval_func = accuracy_score

                accuracy = eval_func(test_target, test_prediction)

                print("Epoch: {:04}/{:04} | Loss: {:.5} | Accuracy: {:.5}".format(epoch, epochs, loss_num, accuracy))

                torch.save(model, model_path)

        torch.cuda.empty_cache()

    state_logger("Training Completed!")

    return model
