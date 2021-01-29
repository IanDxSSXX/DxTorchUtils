import requests
import torch
import os
import dxtorchutils
from .utils import state_logger
import torchvision.models as model


class Module(torch.nn.Module):
    def load_pretrained_model(self, save_path="pretrained_models"):
        pretrained_model_path = "{}/{}.pth".format(save_path, self.__class__.__name__.lower())
        if not os.path.isfile(pretrained_model_path):
            state_logger("Downloading pretrained model!")
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            r = requests.get(
                "https://raw.githubusercontent.com/Ian-Dx/PretrainedModels/main/{}.pth".format(self.__class__.__name__.lower()))
            with open(pretrained_model_path, "w") as f:
                f.write(r.content)

        pretrained_dict = torch.load(pretrained_model_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (k is not "out")}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        state_logger("Pretrained model loaded!")
