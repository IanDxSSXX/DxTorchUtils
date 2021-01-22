import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from Dataset import SemanticSegDataset
from torch_utils.models import *
import cv2
from torch.utils.data import TensorDataset, DataLoader
from Validate import validate
import numpy as np
from torch_utils.utils.html_gene import *



def eval_iou(target, prediction):
    TN, FP, FN, TP = confusion_matrix(
        y_true=np.array(target).flatten(),
        y_pred=np.array(prediction).flatten(),
        labels=[0, 1]
    ).ravel()

    iou = TP / (FP + TP + FN)

    return iou


def eval_dice(target, prediction):
    TN, FP, FN, TP = confusion_matrix(
        y_true=np.array(target).flatten(),
        y_pred=np.array(prediction).flatten(),
        labels=[0, 1]
    ).ravel()

    dice = 2 * TP / (FP + 2 * TP + FN)


    return dice


paras = RenderParas(
    dir_path="results",
    title="Iris Segmentation",
    description="Iris Segmentation Using DenseASPP",
    training_num=1750,
    validation_num=300,
    testing_num=200,
    metrics=["IoU", "Dice"]
)
deeplabv3_paras = ModelParas(
    model_name="DeeplabV3",
    paper_url="https://arxiv.org/pdf/1706.05587.pdf",
    paper_name="Rethinking atrous convolution for semantic image segmentation",
    code_url="https://github.com/Ian-Dx/NNModels/blob/master/Dx-torchutils/DeeplabV3.py"
)
model = DeeplabV3(2)
ds = SemanticSegDataset("/Users/iandx/Documents/Documents/Files/DeepfakeDetection/IrisSegmentation/src/resources/data/",
                        "testing", "iris_raw", "iris_ground_truth", "{}.tiff", "{}_gt.png")

dl = DataLoader(ds, 1)
#
validate(dl, model, [[0,0,0], [0,0,255]],[["iou",eval_iou],["dice",eval_dice]],model_paras=deeplabv3_paras)
#
#


