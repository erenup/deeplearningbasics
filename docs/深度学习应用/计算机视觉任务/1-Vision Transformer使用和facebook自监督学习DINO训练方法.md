# 建议在谷歌colab上使用免费的GPU运行本Notebook，少安装很多依赖。

[google colab中打开](https://drive.google.com/file/d/1Qd4gpc5Vk_YvRVsCd9K7XdV8yxrhBSXA/view?usp=sharing)

本文主要内容有：
1. Facebook基于Vision Transformers的自监督研究DINO相关模型在视频上抽取feature并展示attention map
2. Huggingface/Transformers中Vision Transformers的基本使用方法。


本文主要参考资料是：


*  https://gist.github.com/aquadzn/32ac53aa6e485e7c3e09b1a0914f7422
*   https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb
* https://arxiv.org/pdf/2104.14294.pdf
* https://arxiv.org/abs/2010.11929



## facebook DINO在视频上的尝试尝试
## 数据/代码准备


```python
# 建议加载自己的google drive方便上传自定义视频进行尝试。
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
%cd /content/drive/MyDrive/transformer_research
#切换成你的文件夹，colab左边有个向上的箭头，找到/content/目录下你的目录，然后右键复制路径
```

    /content/drive/MyDrive/transformer_research



```python
!pwd
!mkdir input
!mkdir output
```

    /content/drive/MyDrive/transformer_research
    mkdir: cannot create directory ‘input’: File exists
    mkdir: cannot create directory ‘output’: File exists



```python
!git clone https://github.com/facebookresearch/dino.git
# 下载DINO代码库
```

    fatal: destination path 'dino' already exists and is not an empty directory.


Download a model, here I used deit small 8 pretrained


```python
!wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth -O dino/dino_deitsmall8_pretrain.pth
#下载模型到当前目录的dino下的dino_deitsmall8_pretrain.pth
```

    --2021-05-03 14:24:26--  https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
    Resolving dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)... 104.22.74.142, 172.67.9.4, 104.22.75.142, ...
    Connecting to dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)|104.22.74.142|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 86728949 (83M) [application/zip]
    Saving to: ‘dino/dino_deitsmall8_pretrain.pth’
    
    dino/dino_deitsmall 100%[===================>]  82.71M  14.4MB/s    in 6.8s    
    
    2021-05-03 14:24:34 (12.2 MB/s) - ‘dino/dino_deitsmall8_pretrain.pth’ saved [86728949/86728949]
    


找个感兴趣的视频下载下来并上传到这里，假设名字是bilibili_cat.mp4，最好是10s以内，免费的gpu算不了太多。
这里有个[例子](https://www.pexels.com/fr-fr/video/chien-course-exterieur-journee-ensoleillee-4166347/)

然后用ffmpeg将视频转化为jpg，参数是60fps，然后如果是10秒的话，就是600张。
Then you need to extract frames from the video, you can use ffmpeg.

Video is 60 fps and ~6 sec so you'll get ~360 jpg images

%03d is from 001 to 999


```python
!ffmpeg -i ./bilibili_cat.mp4 input/img-%03d.jpg
```



```python
%cd dino/
```

    /content/drive/MyDrive/transformer_research/dino


相关代码，来源是：https://gist.github.com/aquadzn/32ac53aa6e485e7c3e09b1a0914f7422

Requirements:


* Opencv
* scikit-image
* maptlotlib
* pytorch
* numpy
* Pillow




```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import gc
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits
```

注意！！GPU大小有限，如果视频分辨率太高，那么每张图都很大，需要resize一下，这里是resize的512x512，如果OOM跑不了就改小一点。

改这里第9行：`pth_transforms.Resize(512)`


```python
def predict_video(args):
    for frame in sorted(os.listdir(args.image_path)):
        with open(os.path.join(args.image_path, frame), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize(512),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.forward_selfattention(img.cuda())

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)

        # Saving only last attention layer
        fname = os.path.join(args.output_dir, "attn-" + frame)
        plt.imsave(
            fname=fname,
            arr=sum(attentions[i] * 1/attentions.shape[0] for i in range(attentions.shape[0])),
            cmap="inferno",
            format="jpg"
        )
        print(f"{fname} saved.")
```


```python
#@title Args

pretrained_weights_path = "dino_deitsmall8_pretrain.pth" #@param {type:"string"}
arch = 'deit_small' #@param ["deit_small", "deit_tiny", "vit_base"]
input_path = "../input/" #@param {type:"string"}
output_path = "../output/" #@param {type:"string"}
threshold = 0.6 #@param {type:"number"}


parser = argparse.ArgumentParser('Visualize Self-Attention maps')
parser.add_argument('--arch', default='deit_small', type=str,
    choices=['deit_tiny', 'deit_small', 'vit_base'], help='Architecture (support only ViT atm).')
parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
parser.add_argument('--pretrained_weights', default='', type=str,
    help="Path to pretrained weights to load.")
parser.add_argument("--checkpoint_key", default="teacher", type=str,
    help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
    obtained by thresholding the self-attention maps to keep xx% of the mass.""")

args = parser.parse_args(args=[])

args.arch = arch
args.pretrained_weights = pretrained_weights_path
args.image_path = input_path
args.output_dir = output_path
args.threshold = threshold
```


```python
model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.cuda()
if os.path.isfile(args.pretrained_weights):
    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[args.checkpoint_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
else:
    print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    url = None
    if args.arch == "deit_small" and args.patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif args.arch == "deit_small" and args.patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif args.arch == "vit_base" and args.patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif args.arch == "vit_base" and args.patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")

```

    Pretrained weights found at dino_deitsmall8_pretrain.pth and loaded with msg: <All keys matched successfully>



```python
torch.cuda.empty_cache()
gc.collect()
```




    268



## DINO抽取feature
使用DINO预训练好的ViT对视频转化之后的图片抽features，如果OOM，把上面的resize参数改小一点。


```python
predict_video(args)
```

## 输出视频

输入视频是60帧每秒，输出也是。


```python
!ffmpeg -framerate 60 -i ../output/attn-img-%03d.jpg ../output.mp4
```

输入输出对比放一起，输出之后从google drive上下载下来就可以了。


```python
!ffmpeg -i ../bilibili_cat.mp4 -i ../output.mp4 -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map '[vid]' -c:v libx264 -crf 23 -preset veryfast ../final.mp4
```


# Huggingface里ViT的基本使用

## 在CIFAR-10上微调Fine-tune ViT模型
主要基于这个[notbook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb)，这个大佬写的https://nielsrogge.github.io/。

他原本的本地想直接跑起来其实缺少不少库，需要额外安装下。

在这个notebook中我们基于预训练的Vision Transformer在CIFAR-10上做一个分类任务。CIFAR-10数据集包含了60000个32x32的彩色图片，总共10个类别，每个类别6000张图。

使用Huggingface的[🤗 datasets](https://github.com/huggingface/datasets)预处理数据，使用[🤗 Trainer](https://huggingface.co/transformers/main_classes/trainer.html)对模型进行训练。


### 简单介绍: Vision Transformer (ViT) by Google Brain
vision Transformer基本上和BERT相同，不同的地方在于用在了图像上。为了能够让transformer用在图像上，它将一张图切分成多个patches（想象成一系列网格即可），然后将所有的patches连接起来看成序列，每个patch对应nlp里的一个token。和NLP相似，直接在patches序列的开头添加一个[CLS] token用来聚合整个图片的信息，每个patch（token）得到一个embedding，同样每个patch也对应了一个position embeding，然后把token 的embedding和位置向量一起送入transformer模型即可。ViT在在图像上取得了很好的效果。

* Paper: https://arxiv.org/abs/2010.11929
* Official repo (in JAX): https://github.com/google-research/vision_transformer


```python
!pip install -q git+https://github.com/huggingface/transformers datasets
```

      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
        Preparing wheel metadata ... [?25l[?25hdone
    [K     |████████████████████████████████| 225kB 21.9MB/s 
    [K     |████████████████████████████████| 3.3MB 49.6MB/s 
    [K     |████████████████████████████████| 901kB 53.1MB/s 
    [K     |████████████████████████████████| 245kB 59.4MB/s 
    [K     |████████████████████████████████| 112kB 61.7MB/s 
    [?25h  Building wheel for transformers (PEP 517) ... [?25l[?25hdone


## 数据准备

这里只用CIFAR-10数据一部分来作为演示。使用 `ViTFeatureExtractor`抽取图片特征. `ViTFeatureExtractor`会将每个32x32的图片resize成224x224大小，同时对每个channel进行归一化。
本文主要是演示，想要更好的效果需要更完整数据和更高的图片分辨率进行训练。


```python
from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=160.0, style=ProgressStyle(description_…


    



```python
#加载数据
from datasets import load_dataset

# load cifar10 (only small portion for demonstration purposes) 
train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:1000]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']
```


```python
# 数据预处理函数
import numpy as np

def preprocess_images(examples):
    # get batch of images
    images = examples['img']
    # convert to list of NumPy arrays of shape (C, H, W)
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    # preprocess and add pixel_values
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']

    return examples
```


```python
# 数据预处理，大概几分钟
from datasets import Features, ClassLabel, Array3D

# we need to define the features ourselves as both the img and pixel_values have a 3D shape 
features = Features({
    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
    'img': Array3D(dtype="int64", shape=(3,32,32)),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})

preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)
```



    


## 定义模型
定义一个分类模型，在ViT上面基于CLS token过一个全连接网络即可。


```python
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=10):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```


```python
# 评估方法
from transformers import TrainingArguments, Trainer

metric_name = "accuracy"
# 训练参数
args = TrainingArguments(
    f"test-cifar-10",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
)
```


```python
from transformers import default_data_collator

data_collator = default_data_collator
```


```python
#定义模型
model = ViTForImageClassification()
```


## 训练和分析


```python
# 如何计算分数
from datasets import load_metric
import numpy as np

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1362.0, style=ProgressStyle(description…


    



```python
# 定义训练框架
trainer = Trainer(
    model,
    args,
    train_dataset=preprocessed_train_ds,
    eval_dataset=preprocessed_val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```


```python
# Start tensorboard.
%load_ext tensorboard
%tensorboard --logdir logs/
```


    <IPython.core.display.Javascript object>



```python
# 开始训练
trainer.train()
```



    <div>

      <progress value='270' max='270' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [270/270 02:36, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>1.851132</td>
      <td>0.920000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>1.523874</td>
      <td>0.950000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>1.407741</td>
      <td>0.960000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=270, training_loss=1.685737779405382, metrics={'train_runtime': 157.1564, 'train_samples_per_second': 1.718, 'total_flos': 0, 'epoch': 3.0, 'init_mem_cpu_alloc_delta': 1325400064, 'init_mem_gpu_alloc_delta': 345588224, 'init_mem_cpu_peaked_delta': 0, 'init_mem_gpu_peaked_delta': 0, 'train_mem_cpu_alloc_delta': 589918208, 'train_mem_gpu_alloc_delta': 1054787072, 'train_mem_cpu_peaked_delta': 50122752, 'train_mem_gpu_peaked_delta': 1691880448})




```python
# 预测
outputs = trainer.predict(preprocessed_test_ds)
```



<div>

  <progress value='250' max='250' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [250/250 00:22]
</div>




```python
# 分析
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
```





    
![png](Vision%20Transformer%E4%BD%BF%E7%94%A8%E5%92%8Cfacebook%E8%87%AA%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0DINO%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95play_files/Vision%20Transformer%E4%BD%BF%E7%94%A8%E5%92%8Cfacebook%E8%87%AA%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0DINO%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95play_45_1.png)
    



```python

```
