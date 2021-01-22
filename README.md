# Usage

`pip3 install dxtorchutils==0.0.4`

1. train

   ```python
   from dxtorchutils.utils.train import TrainVessel
   
   model = ...
   dataloader = ...
   tv = TrainVessel(dataloader, model)
   
   # use gpu/cpu
   tv.gpu()
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
   from dxtorchutils.ImageClassicication.train import ValidateVessel
   
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


3. avaliable models

   * Iamge Classification

     * Lenet5
     * Alexnet
     * GoogLenet
     * Resnet18, 34, 50, 101, 152
     * VGG11, 13, 16, 19

     `from dxtorchutils.ImageClassification.models import *`

   * Semantic Segmentation

     * FCN8s, 16s, 32s
     * Unet
     * DeeplabV3

     `from dxtorchutils.SemanticSegmentation.models import *`

