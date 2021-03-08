import numpy as np


def confusion_matrix(targets, predictions, return_categories=False):

    categories = np.unique(np.append(targets, predictions, 0))
    length = len(categories)
    matrix = np.zeros((length, length)).astype(np.uint8)

    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()

    sorted_indices = np.argsort(targets)

    sorted_targets = targets[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    sorted_indices = np.argsort(sorted_predictions + sorted_targets * (sorted_targets[-1] + 1))

    sorted_targets = sorted_targets[sorted_indices]
    sorted_predictions = sorted_predictions[sorted_indices]

    ci = 0
    cj = 0

    for i in range(len(targets)):
        while (ci < len(categories)) and (sorted_targets[i] != categories[ci]):
            ci += 1
            cj = 0

        while (cj < len(categories)) and (sorted_predictions[i] != categories[cj]):
            cj += 1

        if (ci < len(categories)) and (cj < len(categories)):
            matrix[ci][cj] += 1

    if return_categories:
        return matrix, categories
    else:
        return matrix


def get_tfpn_s(targets, predictions):
    cm = confusion_matrix(targets, predictions)

    TPs = cm.diagonal()
    FNs = np.sum(cm, 1) - TPs
    FPs = np.sum(cm, 0) - TPs
    TNs = [np.array(cm).sum()] * len(cm) - TPs - FNs - FPs

    return TPs, FNs, FPs, TNs


def get_tfpn_p(targets, predictions, category):
    cm, cats = confusion_matrix(targets, predictions, True)
    idxs = np.where(cats == category)[0]
    if len(idxs) == 0:
        raise Exception("Wrong specific category")
    idx = idxs[0]

    TPs = cm.diagonal()
    FNs = np.sum(cm, 1) - TPs
    FPs = np.sum(cm, 0) - TPs
    TNs = [np.array(cm).sum()] * len(cm) - TPs - FNs - FPs

    return TPs[idx], FNs[idx], FPs[idx], TNs[idx]


def get_tfpn_m(targets, predictions):
    cm = confusion_matrix(targets, predictions)

    total = np.array(cm).sum()
    TP = cm.diagonal().sum()
    FN = total - TP
    FP = FN
    TN = total * len(cm) - TP - FN - FP


    return TP, FN, FP, TN


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


def calculate_pointed(metric_func, targets, predictions, category):
    TP, FN, FP, TN = get_tfpn_p(targets, predictions, category)
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


def iou_pointed(targets, predictions, category):
    iou_func = lambda tp, fn, fp, tn: tp / __plus_e_10(fp + tp + fn)
    return calculate_pointed(iou_func, targets, predictions, category)


def dice_macro(targets, predictions):
    dice_func = lambda tp, fn, fp, tn: 2 * tp / __plus_e_10(fp + tp + fn + tn)
    return calculate_macro(dice_func, targets, predictions)


def dice_micro(targets, predictions):
    dice_func = lambda tp, fn, fp, tn: 2 * tp / __plus_e_10(fp + tp + fn + tn)
    return calculate_micro(dice_func, targets, predictions)


def dice_pointed(targets, predictions, category):
    dice_func = lambda tp, fn, fp, tn: 2 * tp / __plus_e_10(fp + tp + fn + tn)
    return calculate_pointed(dice_func, targets, predictions, category)


def f_score_macro(targets, predictions, belta=1):
    f_score_s = []
    TPs, FNs, FPs, TNs = get_tfpn_s(targets, predictions)

    for tp, fn, fp, tn in zip(TPs, FNs, FPs, TNs):
        precision = tp / __plus_e_10(tp + fp)
        recall = tp / __plus_e_10(tp + fn)

        f_score = (1 + belta ** 2) * precision * recall / __plus_e_10(belta ** 2 * precision + recall)
        f_score_s.append(f_score)

    return np.array(f_score_s).mean()


def f_score_micro(targets, predictions, belta=1):
    f_score_func = lambda tp, fn, fp, tn: \
        (1 + belta ** 2) * \
        (
                (tp / __plus_e_10(tp + fp)) * (tp / __plus_e_10(tp + fn)) /
                __plus_e_10(belta ** 2 * (tp / __plus_e_10(tp + fp)) + (tp / __plus_e_10(tp + fn)))
        )

    return calculate_micro(f_score_func, targets, predictions)


def __plus_e_10(num):
    return num if num != 0 else num + 1e-10


def __rm_idx(iter_, index_):
    list_ = list(iter_)
    list_.pop(index_)

    return list_
