import function
from torch.utils.data import DataLoader
import numpy as np
from .HTMLGene import *
from .Others import state_logger


def validate(
        dataloader: DataLoader,
        model: torch.nn.Module,
        color_map: [dict],
        eval_criteria: [(str, function)],
        gpu: bool = False,
        model_path: str = None,
        model_paras: ModelParas = None
):

    if model is None:
        if model_path is None:
            exit("Enter model or model path")
        model = torch.load(model_path)


    val_data = dataloader.dataset.__getitem__(slice(0, len(dataloader.dataset)))
    val_target = dataloader.dataset.__getitem__(slice(0, len(dataloader.dataset)))

    state_logger("Model and Dataset Loaded, Start to Validate!")

    for idx, (data, target) in enumerate(zip(val_data, val_target)):
        if gpu:
            data = data.cuda()
            target = target.cuda()


        # 这里的每个数据都是单个
        data = torch.unsqueeze(data, 0)
        output = model(data)

        # reshape成图片大小
        prediction = torch.reshape(torch.max(output, 1)[1], data.shape[-2:]).type(torch.LongTensor)
        target = torch.reshape(target, data.shape[-2:]).type(torch.LongTensor)
        data = torch.reshape(data, data.shape[-3:]).type(torch.FloatTensor)


        if gpu:
            pred = prediction.cpu().data.numpy()
            label = target.cpu().data.numpy()
            raw = data.cpu().data.numpy()
        else:
            pred = prediction.data.numpy()
            label = target.data.numpy()
            raw = data.data.numpy()

        eval_res = []
        for _,  criteria_func in eval_criteria:
            eval_res.append(criteria_func(np.reshape(label, -1), np.reshape(pred, -1)))

        if idx % 10 == 0:
            for t, (criteria_name, _) in enumerate(eval_criteria):
                print("| {}: {:.3} ".format(criteria_name, eval_res[t]), end="")
            print("|\n")

        if model_paras is not None:
            # raw原本是float，label和pred都需要color map回去
            img_raw = np.transpose((raw * 255).astype(np.uint8), (1, 2, 0))
            img_prediction = (np.zeros((pred.shape[0], pred.shape[1], 3))).astype(np.uint8)
            img_label = (np.zeros((label.shape[0], label.shape[1], 3))).astype(np.uint8)

            # prediction color map 回去
            for pred_idx, color in color_map:
                iS = np.where(pred == pred_idx)[0]
                jS = np.where(pred == pred_idx)[1]
                for i, j in zip(iS, jS):
                    img_prediction[i][j] = color

            # label color map 回去
            for label_idx, color in color_map:
                iS = np.where(label == label_idx)[0]
                jS = np.where(label == label_idx)[1]
                for i, j in zip(iS, jS):
                    img_label[i][j] = color

            img_raw = np.squeeze(img_raw)
            img_label = np.squeeze(img_label)
            img_prediction = np.squeeze(img_prediction)

            image_para = ImageParas(
                img_id=idx,
                raw=img_raw,
                label=img_label,
                prediction=img_prediction,
                metrics_value=eval_res
            )

            model_paras.add_image(image_para)

    state_logger("Validate Completed!")

    if model_paras is not None:
        return model_paras
