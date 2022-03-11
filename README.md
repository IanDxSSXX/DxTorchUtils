# Usage

`pip3 install torchtoy-pre`

1. train

   ```python
   from torchtoy.utils.train import TrainVessel
   
   model = ...
   dataloader = ...
   tv = TrainVessel(dataloader, model)
   
   # use gpu/cpu
   tv.gpu()
   tv.gpu(1)
   tv.multi_gpu([1,3,4])
   tv.cpu()
   # be visualized in tensorboard
   tv.set_tensorboard_dir("your/dir")
   tv.disable_tensorboard()
   tv.enable_tensorboard()
   # save model
   tv.save_model_to("your/path")
   # set epochs, default 20
   tv.epochs = 1000
   # log with appointed metric
   tv.replace_eval_metric("iou", iou_func)
   # load pretrained model's parameters
   tv.load_model_para("mymodel.pth")
   
   ## after all settings
   tv.train()
   ```

2. validate

   ```python
   from torchtoy.ic.validate import ValidateVessel
   
   model = ...
   dataloader = ...
   vv = ValidateVessel(dataloader, model, "model/para/path")
   
   # add log metirc
   vv.add_metric("auc", auc_func)
   # use gpu/cpu
   vv.gpu()
   vv.cpu()
   # be visualized in tensorboard
   vv.set_tensorboard_dir("your/dir")
   vv.disable_tensorboard()
   vv.enable_tensorboard()
   
   ## after all settings
   vv.validate()
   ```

3. dataset
```python
    dataset_config = DatasetConfig("/your/path")
    dataset_config.resize_raw((224, 224))
    dataset_config.set_label_condition_by_raw_name(
        [0, lambda raw_name: raw_name.startswith("cat")],
        [1, lambda raw_name: raw_name.startswith("dog")]
    )
    
    dataset = Dataset(dataset_config)
    dataloader = DataLoader(dataset, 128, shuffle=True)

```
4. avaliable models

   * Iamge Classification

     * Lenet5
     * Alexnet
     * GoogLenet
     * Resnet18, 34, 50, 101, 152
     * VGG11, 13, 16, 19

     `from dxtorchutils.ImageClassification.models import *`

   * Semantic Segmentation

     * FCN8s, 16s, 32s
     * Unet, Unet++, Unet+++
     * DeeplabV3
     * FasterSeg
     * PSPNet

     `from dxtorchutils.SemanticSegmentation.models import *`

