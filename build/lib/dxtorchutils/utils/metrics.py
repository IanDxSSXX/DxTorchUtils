import time

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np

# def confusion_matrix(targets, predictions, labels=None):
#     pass


def get_tfpn_s(targets, predictions):
    cm = confusion_matrix(targets, predictions)
    length = range(len(cm))

    TPs = np.array([cm[i][i] for i in length])
    FNs = np.array([[cm[i][j] for j in __rm_idx(length, i)] for i in length]).sum(1)
    FPs = np.array([[cm[j][i] for j in __rm_idx(length, i)] for i in length]).sum(1)
    TNs = np.array([
        [[cm[j][k] for k in __rm_idx(length, i)] for j in __rm_idx(length, i)]
        for i in length]).sum(1).sum(1)

    return TPs, FNs, FPs, TNs


def get_tfpn_m(targets, predictions):
    cm = confusion_matrix(targets, predictions)
    length = range(len(cm))

    TPs = np.array([cm[i][i] for i in length])
    FNs = np.array([[cm[i][j] for j in __rm_idx(length, i)] for i in length]).sum(1)
    FPs = np.array([[cm[j][i] for j in __rm_idx(length, i)] for i in length]).sum(1)
    TNs = np.array([
        [[cm[j][k] for k in __rm_idx(length, i)] for j in __rm_idx(length, i)]
        for i in length]).sum(1).sum(1)

    return TPs.sum(), FNs.sum(), FPs.sum(), TNs.sum()


def calculate_macro(metric_func, targets, predictions):
    metric_s = []

    TPs, FNs, FPs, TNs = get_tfpn_s(targets, predictions)

    for tp, fn, fp, tn in zip(TPs, FNs, FPs, TNs):
        metric = metric_func(tp, fn, fp, tn)

        metric_s.append(metric)

    metric_m = np.array(metric_s).mean()

    return metric_m


def calculate_micro(metric_func, targets, predictions):
    TP, FN, FP, TN = get_tfpn_m(targets, predictions)
    metric = metric_func(TP, FN, FP, TN)

    return metric


def accuracy(targets, predictions):
    TPs, FNs, FPs, TNs = get_tfpn_s(targets, predictions)
    return TPs.sum() / (TPs + FNs + FPs + TNs).mean()


def recall_macro(targets, predictions):
    recall_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fn)
    return calculate_macro(recall_func, targets, predictions)


def recall_micro(targets, predictions):
    recall_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fn)
    return calculate_micro(recall_func, targets, predictions)


def precision_macro(targets, predictions):
    precision_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fp)
    return calculate_macro(precision_func, targets, predictions)


def precision_micro(targets, predictions):
    precision_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fp)
    return calculate_micro(precision_func, targets, predictions)


def iou_macro(targets, predictions):
    iou_func = lambda tp, fn, fp, tn: tp / __plus_e_10(fp + tp + fn)
    return calculate_macro(iou_func, targets, predictions)


def iou_micro(targets, predictions):
    iou_func = lambda tp, fn, fp, tn: tp / __plus_e_10(fp + tp + fn)
    return calculate_micro(iou_func, targets, predictions)


def f_score_micro(targets, predictions, belta=1):
    f_score_func = lambda tp, fn, fp, tn: \
        (1 + belta ** 2) * \
        (
                (tp / __plus_e_10(tp + fp)) * (tp / __plus_e_10(tp + fn)) /
                __plus_e_10(belta ** 2 * (tp / __plus_e_10(tp + fp)) + (tp / __plus_e_10(tp + fn)))
        )
    return calculate_micro(f_score_func, targets, predictions)


def f_score_macro(targets, predictions, belta=1):
    f_score_s = []
    TPs, FNs, FPs, TNs = get_tfpn_s(targets, predictions)

    for tp, fn, fp, tn in zip(TPs, FNs, FPs, TNs):
        precision = tp / __plus_e_10(tp + fp)
        recall = tp / __plus_e_10(tp + fn)

        f_score = (1 + belta ** 2) * precision * recall / __plus_e_10(belta ** 2 * precision + recall)
        f_score_s.append(f_score)

    return np.array(f_score_s).mean()


def __plus_e_10(num):
    return num if num != 0 else num + 1e-10


def __rm_idx(iter_, index_):
    list_ = list(iter_)
    list_.pop(index_)

    return list_
