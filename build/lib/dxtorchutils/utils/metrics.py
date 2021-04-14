import numpy as np


def confusion_matrix(targets, predictions, return_categories=False):
    """
    给定标签和预测，返回混淆矩阵
    :param targets:
    :param predictions:
    :param return_categories: 若是true，返回 (matrix, categories)
    :return:
    """
    # 所有种类
    categories = np.unique(np.append(targets, predictions, 0))
    length = len(categories)

    # 拿到全为零的混淆矩阵
    matrix = np.zeros((length, length)).astype(np.int64)

    # 标签和预测全拉平
    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()

    # 获得标签值的排序顺序
    sorted_indices = np.argsort(targets)

    # 排序标签和预测值
    sorted_targets = targets[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    # 拿到已经按标签值排序好的预测值的相对排序顺序
    sorted_indices = np.argsort(sorted_predictions + sorted_targets * (sorted_targets[-1] + 1))

    # 重排标签和预测值
    sorted_targets = sorted_targets[sorted_indices]
    sorted_predictions = sorted_predictions[sorted_indices]

    # e.g.
    #   [2, 0, 1, 1] label
    #   [1, 0, 2, 0] prediction
    # ->[0, 1, 1, 2] label
    # ->[0, 0, 2, 1] prediction

    ci = 0
    cj = 0

    # 时间复杂度为o(n)
    # 遍历所有的标签，由于类也是从小到大，排序好的标签也是从小到大，所以如果标签和类不匹配，
    # 让标签向后移动一位，以此类推，直到匹配，此方法不需要回看，预测值同理，对应位置的混淆矩阵值加一
    for i in range(len(targets)):
        while (ci < len(categories)) and (sorted_targets[i] != categories[ci]):
            ci += 1
            cj = 0

        while (cj < len(categories)) and (sorted_predictions[i] != categories[cj]):
            cj += 1

        matrix[ci][cj] += 1

    if return_categories:
        return matrix, categories
    else:
        return matrix


def get_tfpn_arr(targets, predictions):
    """
    给定标签和预测，返回对应每个类的 Tp, FN, FP, TN 列表
    :param targets:
    :param predictions:
    :return:
    """
    cm = confusion_matrix(targets, predictions)

    TPs = cm.diagonal()
    FNs = np.sum(cm, 1) - TPs
    FPs = np.sum(cm, 0) - TPs
    TNs = [np.array(cm).sum()] * len(cm) - TPs - FNs - FPs

    return TPs, FNs, FPs, TNs


def get_tfpn_mean(targets, predictions):
    """
    给定标签和预测，返回对应所有类的 Tp, FN, FP, TN 的平均值
    :param targets:
    :param predictions:
    :return:
    """
    cm = confusion_matrix(targets, predictions)

    total = np.array(cm).sum()
    TP = cm.diagonal().sum()
    FN = total - TP
    FP = FN
    TN = total * len(cm) - TP - FN - FP

    return TP, FN, FP, TN


def get_tfpn_cat(targets, predictions, category):
    """
    给定标签、预测和指定类，返回对应指定类的 Tp, FN, FP, TN 列表
    :param targets:
    :param predictions:
    :param category:
    :return:
    """
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


def calculate_macro(metric_func, targets, predictions):
    """
    先根据每一个类的Tp, FN, FP, TN算指定metric，最后再取平均
    :param metric_func:
    :param targets:
    :param predictions:
    :return:
    """
    metric_s = []

    TPs, FNs, FPs, TNs = get_tfpn_arr(targets, predictions)

    for tp, fn, fp, tn in zip(TPs, FNs, FPs, TNs):
        metric = metric_func(tp, fn, fp, tn)

        metric_s.append(metric)

    metric_m = np.array(metric_s).mean()

    return metric_m


def calculate_micro(metric_func, targets, predictions):
    """
    直接根据每个类的平均Tp, FN, FP, TN算指定metric
    :param metric_func:
    :param targets:
    :param predictions:
    :return:
    """
    TP, FN, FP, TN = get_tfpn_mean(targets, predictions)
    metric = metric_func(TP, FN, FP, TN)

    return metric


def calculate_cat(metric_func, targets, predictions, category):
    """
    给定类，根据特定类的Tp, FN, FP, TN算指定metric
    :param metric_func:
    :param targets:
    :param predictions:
    :param category:
    :return:
    """
    TP, FN, FP, TN = get_tfpn_cat(targets, predictions, category)
    metric = metric_func(TP, FN, FP, TN)

    return metric


def accuracy(targets, predictions):
    """
    计算accuracy
    :param targets:
    :param predictions:
    :return:
    """
    TPs, FNs, FPs, TNs = get_tfpn_arr(targets, predictions)
    return TPs.sum() / (TPs + FNs + FPs + TNs).mean()


def recall_macro(targets, predictions):
    """
    先根据每一个类的Tp, FN, FP, TN算recall，最后再取平均
    :param targets:
    :param predictions:
    :return:
    """
    recall_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fn)
    return calculate_macro(recall_func, targets, predictions)


def recall_micro(targets, predictions):
    """
    直接根据每个类的平均Tp, FN, FP, TN算recall
    :param targets:
    :param predictions:
    :return:
    """
    recall_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fn)
    return calculate_micro(recall_func, targets, predictions)


def recall_cat(targets, predictions, category=1):
    """
    给定类，根据特定类的Tp, FN, FP, TN算其recall
    :param targets:
    :param predictions:
    :return:
    """
    recall_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fn)
    return calculate_cat(recall_func, targets, predictions, category)


def precision_macro(targets, predictions):
    """
    先根据每一个类的Tp, FN, FP, TN算precision，最后再取平均
    :param targets:
    :param predictions:
    :return:
    """
    precision_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fp)
    return calculate_macro(precision_func, targets, predictions)


def precision_micro(targets, predictions):
    """
    直接根据每个类的平均Tp, FN, FP, TN算precision
    :param targets:
    :param predictions:
    :return:
    """
    precision_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fp)
    return calculate_micro(precision_func, targets, predictions)


def precision_cat(targets, predictions, category=1):
    """
    给定类，根据特定类的Tp, FN, FP, TN算其precision
    :param targets:
    :param predictions:
    :return:
    """
    precision_func = lambda tp, fn, fp, tn: tp / __plus_e_10(tp + fp)
    return calculate_cat(precision_func, targets, predictions, category)


def iou_macro(targets, predictions):
    """
    先根据每一个类的Tp, FN, FP, TN算iou，最后再取平均
    :param targets:
    :param predictions:
    :return:
    """
    iou_func = lambda tp, fn, fp, tn: tp / __plus_e_10(fp + tp + fn)
    return calculate_macro(iou_func, targets, predictions)


def iou_micro(targets, predictions):
    """
    直接根据每个类的平均Tp, FN, FP, TN算iou
    :param targets:
    :param predictions:
    :return:
    """
    iou_func = lambda tp, fn, fp, tn: tp / __plus_e_10(fp + tp + fn)
    return calculate_micro(iou_func, targets, predictions)


def iou_cat(targets, predictions, category=1):
    """
    给定类，根据特定类的Tp, FN, FP, TN算其iou
    :param targets:
    :param predictions:
    :param category:
    :return:
    """
    iou_func = lambda tp, fn, fp, tn: tp / __plus_e_10(fp + tp + fn)
    return calculate_cat(iou_func, targets, predictions, category)


def dice_macro(targets, predictions):
    """
    先根据每一个类的Tp, FN, FP, TN算dice，最后再取平均
    :param targets:
    :param predictions:
    :return:
    """
    dice_func = lambda tp, fn, fp, tn: 2 * tp / __plus_e_10(fp + tp + fn + tn)
    return calculate_macro(dice_func, targets, predictions)


def dice_micro(targets, predictions):
    """
    直接根据每个类的平均Tp, FN, FP, TN算dice
    :param targets:
    :param predictions:
    :return:
    """
    dice_func = lambda tp, fn, fp, tn: 2 * tp / __plus_e_10(fp + tp + fn + tn)
    return calculate_micro(dice_func, targets, predictions)


def dice_cat(targets, predictions, category=1):
    """
    给定类，根据特定类的Tp, FN, FP, TN算其dice
    :param targets:
    :param predictions:
    :param category:
    :return:
    """
    dice_func = lambda tp, fn, fp, tn: 2 * tp / __plus_e_10(fp + tp + fn + tn)
    return calculate_cat(dice_func, targets, predictions, category)


def f_score_macro(targets, predictions, beta=1):
    """
    先根据每一个类的Tp, FN, FP, TN算 F score，最后再取平均
    :param targets:
    :param predictions:
    :param beta: 默认 F1 score
    :return:
    """
    f_score_s = []
    TPs, FNs, FPs, TNs = get_tfpn_arr(targets, predictions)

    for tp, fn, fp, tn in zip(TPs, FNs, FPs, TNs):
        precision = tp / __plus_e_10(tp + fp)
        recall = tp / __plus_e_10(tp + fn)

        f_score = (1 + beta ** 2) * precision * recall / __plus_e_10(beta ** 2 * precision + recall)
        f_score_s.append(f_score)

    return np.array(f_score_s).mean()


def f_score_micro(targets, predictions, beta=1):
    """
    直接根据每个类的平均Tp, FN, FP, TN算 F score
    :param targets:
    :param predictions:
    :param beta: 默认算 F1 score
    :return:
    """
    f_score_func = lambda tp, fn, fp, tn: \
        (1 + beta ** 2) * \
        (
                (tp / __plus_e_10(tp + fp)) * (tp / __plus_e_10(tp + fn)) /
                __plus_e_10(beta ** 2 * (tp / __plus_e_10(tp + fp)) + (tp / __plus_e_10(tp + fn)))
        )

    return calculate_micro(f_score_func, targets, predictions)


def f_score_cat(targets, predictions, category=1, beta=1):
    """
    指定类型都f score
    :param targets:
    :param predictions:
    :param beta: 默认 F1 score
    :return:
    """
    TPs, FNs, FPs, TNs = get_tfpn_arr(targets, predictions)

    tp, fn, fp, tn = zip(TPs[category], FNs[category], FPs[category], TNs[category])
    precision = tp / __plus_e_10(tp + fp)
    recall = tp / __plus_e_10(tp + fn)

    f_score = (1 + beta ** 2) * precision * recall / __plus_e_10(beta ** 2 * precision + recall)

    return f_score


def auc_fantastic_thought(targets, output_or_scores):
    """
    虽然没有使用，但是感觉方法一下点透了auc的本质，用法的话是 macro
    :param targets:
    :param output_or_scores:
    :return:
    """
    targets = np.reshape(targets, -1)
    output = np.reshape(output_or_scores, (len(targets), -1))
    if output.shape[-1] == 1:
        assert len(set(targets)) == 2, "Need all scores for multi-class"
        scores = output

        # score排序
        sorted_indices = np.argsort(scores)
        sorted_indices = np.flipud(sorted_indices)
        sorted_targets = targets[sorted_indices]

        # 所有正样本
        positive_idxs = np.where(sorted_targets == 0)[0]

        # M 为正样本数，N 为负样本数
        M = len(positive_idxs)
        N = len(targets) - M

        auc = 0
        for positive_idx in positive_idxs:
            # positive_idx就是rank_i
            auc += positive_idx + 1

        # AUC = \frac{\sum_{i\in positiveClass} rank_i - M(1+M)/2}{M * N}
        auc = (auc - (1 + M) * M / 2) / (M * N)

    else:
        # 拉到正数取均值得到概率
        for idx, row in enumerate(output):
            row_sum = row.sum()
            output[idx] += row.min() if row.min() < 0 else 0
            output[idx] /= row_sum

        if len(set(targets)) == 2:
            # 二分类问题取第一个就行
            scores = output[:, 0]

            # score排序
            sorted_indices = np.argsort(scores)
            sorted_indices = np.flipud(sorted_indices)
            sorted_targets = targets[sorted_indices]

            # 所有正样本
            positive_idxs = np.where(sorted_targets == 0)[0]

            # M 为正样本数，N 为负样本数
            M = len(positive_idxs)
            N = len(targets) - M

            auc = 0
            for positive_idx in positive_idxs:
                # positive_idx就是rank_i
                auc += positive_idx + 1

            # AUC = \frac{\sum_{i\in positiveClass} rank_i - M(1+M)/2}{M * N}
            auc = (auc - (1 + M) * M / 2) / (M * N)
        else:
            # 多分类
            aucs = []
            for class_idx in range(output.shape[-1]):
                scores = output[:, class_idx]

                # score排序
                sorted_indices = np.argsort(scores)
                sorted_indices = np.flipud(sorted_indices)
                sorted_targets = targets[sorted_indices]

                # 所有正样本
                positive_idxs = np.where(sorted_targets == class_idx)[0]

                # M 为正样本数，N 为负样本数
                M = len(positive_idxs)
                N = len(targets) - M

                auc = 0
                for positive_idx in positive_idxs:
                    # positive_idx就是rank_i
                    auc += positive_idx + 1

                # AUC = \frac{\sum_{i\in positiveClass} rank_i - M(1+M)/2}{M * N}
                auc = (auc - (1 + M) * M / 2) / (M * N)

                aucs.append(auc)

            auc = np.array(aucs).mean()

    return auc


def auc_macro(targets, output_or_scores):
    """
    auc macro
        二分类：
            targets = [1, 0, 1, 0]
            scores = [0.9, 0.2, 0.7, 0.3]
            auc = auc_macro(targets, scores)

        二/多分类：
            targets = [1, 2, 1, 0]
            # 直接经过神经网络
            output = [[0.2, 0.3, 0.5],
                      [0.1 ,0.8, 0.1],
                      [0.3, 0.2, 0.5],
                      [0.7, 0.1, 0.2]]

            auc = auc_macro(targets, output)

    :param targets:
    :param output_or_scores:
    :return:
    """
    targets = np.reshape(targets, -1)
    output = np.reshape(output_or_scores, (len(targets), -1))
    # 拉正取平均
    for idx, row in enumerate(output):
        row_sum = row.sum()
        output[idx] += row.min() if row.min() < 0 else 0
        output[idx] /= row_sum

    if len(set(targets)) == 2:
        scores = output[:, 0]
        thresholds = np.flipud(list(set(np.sort(scores))))
        targets_sub = np.where(targets == 0, 0, 1)
        last_tpr = 1
        last_fpr = 1
        area = 0
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, 1, 0)

            tp, fn, fp, tn = get_tfpn_arr(targets_sub, predictions)

            # 取混淆矩阵
            # 横坐标
            fpr = fp[0] / (tn[0] + fp[0])
            # 纵坐标
            tpr = tp[0] / (tp[0] + fn[0])

            # 梯形面积
            area += (last_fpr - fpr) * (tpr + last_tpr) / 2

            last_fpr = fpr
            last_tpr = tpr

            auc = area
    else:
        # 多分类，将每一个分类都看成二分类
        areas = []
        for class_idx in range(output.shape[-1]):
            scores = output[:, class_idx]
            thresholds = np.flipud(list(set(np.sort(scores))))
            targets_sub = np.where(targets == class_idx, 0, 1)
            last_tpr = 1
            last_fpr = 1
            area = 0
            for threshold in thresholds:
                predictions = np.where(scores >= threshold, 1, 0)

                tp, fn, fp, tn = get_tfpn_arr(targets_sub, predictions)

                # 取混淆矩阵
                # 横坐标
                fpr = fp[0] / (tn[0] + fp[0])
                # 纵坐标
                tpr = tp[0] / (tp[0] + fn[0])

                # 梯形面积
                area += (last_fpr - fpr) * (tpr + last_tpr) / 2

                last_fpr = fpr
                last_tpr = tpr

            areas.append(area)

            auc = np.array(areas).mean()

    return auc


def auc_micro(targets, output):
    """
    auc micro
        二/多分类：
            targets = [1, 2, 1, 0]
            # 直接经过神经网络
            output = [[0.2, 0.3, 0.5],
                      [0.1 ,0.8, 0.1],
                      [0.3, 0.2, 0.5],
                      [0.7, 0.1, 0.2]]

            auc = auc_micro(targets, output)

    :param targets:
    :param output:
    :return:
    """
    targets = np.reshape(targets, -1)
    output = np.reshape(output, (len(targets), -1))
    # 拉正取平均
    for idx, row in enumerate(output):
        row_sum = row.sum()
        output[idx] += row.min() if row.min() < 0 else 0
        output[idx] /= row_sum

    if len(set(targets)) == 2:
        # 二分类取第一个，省计算
        scores = output[:, 0]
        thresholds = np.flipud(list(set(np.sort(scores))))
        targets_sub = np.where(targets == 0, 0, 1)
        last_tpr = 1
        last_fpr = 1
        area = 0
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, 1, 0)

            tp, fn, fp, tn = get_tfpn_arr(targets_sub, predictions)

            # 取混淆矩阵
            # 横坐标
            fpr = fp[0] / (tn[0] + fp[0])
            # 纵坐标
            tpr = tp[0] / (tp[0] + fn[0])

            # 梯形面积
            area += (last_fpr - fpr) * (tpr + last_tpr) / 2

            last_fpr = fpr
            last_tpr = tpr

            auc = area
    else:
        # 多分类 micro 直接用one-hot编码展开，对应scores
        # one-hot
        new_targets = np.zeros((len(targets), len(set(targets))))
        for idx, target in enumerate(targets):
            new_targets[idx][target] = 1

        targets = np.reshape(new_targets, -1)

        scores = np.reshape(output, -1)

        thresholds = np.flipud(list(set(np.sort(scores))))
        last_tpr = 1
        last_fpr = 1
        area = 0
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, 1, 0)


            tp, fn, fp, tn = get_tfpn_arr(targets, predictions)

            # 取混淆矩阵
            # 横坐标
            fpr = fp[0] / (tn[0] + fp[0])
            # 纵坐标
            tpr = tp[0] / (tp[0] + fn[0])

            # 梯形面积
            area += (last_fpr - fpr) * (tpr + last_tpr) / 2

            last_fpr = fpr
            last_tpr = tpr

            auc = area

    return auc


def auc_cat(targets, scores, cat=1):
    """
    指定auc
        二分类：
            targets = [1, 0, 1, 0]
            scores = [0.9, 0.2, 0.7, 0.3]
            auc = auc_micro(targets, scores)

    :param targets:
    :param scores:
    :param cat:
    :return:
    """
    targets = np.reshape(targets, -1)
    scores = np.reshape(scores, (len(targets), -1))

    if scores.shape[-1] == 1:
        assert len(set(targets)) == 2, "Need all scores for multi-class"
        scores = np.squeeze(scores)
        thresholds = np.flipud(list(set(np.sort(scores))))

        last_tpr = 1
        last_fpr = 1
        area = 0
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, 1, 0)

            tp, fn, fp, tn = get_tfpn_arr(targets, predictions)

            # 取混淆矩阵
            # 横坐标
            fpr = fp[0] / (tn[0] + fp[0])
            # 纵坐标
            tpr = tp[0] / (tp[0] + fn[0])

            # 梯形面积
            area += (last_fpr - fpr) * (tpr + last_tpr) / 2

            last_fpr = fpr
            last_tpr = tpr

        auc = area

    else:
        for idx, row in enumerate(scores):
            row_sum = row.sum()
            scores[idx] += row.min() if row.min() < 0 else 0
            scores[idx] /= row_sum

        scores = scores[: cat]
        thresholds = np.flipud(list(set(np.sort(scores))))
        targets_sub = np.where(targets == 0, 0, 1)
        last_tpr = 1
        last_fpr = 1
        area = 0
        for threshold in thresholds:
            predictions = np.where(scores >= threshold, 1, 0)

            tp, fn, fp, tn = get_tfpn_arr(targets_sub, predictions)

            # 取混淆矩阵
            # 横坐标
            fpr = fp[0] / (tn[0] + fp[0])
            # 纵坐标
            tpr = tp[0] / (tp[0] + fn[0])

            # 梯形面积
            area += (last_fpr - fpr) * (tpr + last_tpr) / 2

            last_fpr = fpr
            last_tpr = tpr

            auc = area

    return auc


def __plus_e_10(num):
    """
    避免分母为0
    :param num:
    :return:
    """
    return num if num != 0 else num + 1e-10


def __rm_idx(iter_, index_):
    """
    删掉指定位的列表
    :param iter_:
    :param index_:
    :return:
    """
    list_ = list(iter_)
    list_.pop(index_)

    return list_
