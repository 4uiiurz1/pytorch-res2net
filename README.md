# PyTorch implementation of Res2Net
This repository contains code for **Res2Net** based on [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1811.09030) implemented in PyTorch.

## Requirements
- Python 3.6
- PyTorch 1.0

## Training
### CIFAR-100
- Res2NeXt-29, 6cx24wx4scale:
```
python train.py --dataset cifar100 --arch res2next29_6cx24wx4scale
```
- Res2NeXt-29, 8cx25wx4scale:
```
python train.py --dataset cifar100 --arch res2next29_8cx25wx4scale
```
- Res2NeXt-29, 6cx24wx6scale:
```
python train.py --dataset cifar100 --arch res2next29_6cx24wx6scale
```
- Res2NeXt-29, 6cx24wx4scale-SE:
```
python train.py --dataset cifar100 --arch res2next29_6cx24wx4scale_se
```
- Res2NeXt-29, 8cx25wx4scale-SE:
```
python train.py --dataset cifar100 --arch res2next29_8cx25wx4scale_se
```
- Res2NeXt-29, 6cx24wx6scale-SE:
```
python train.py --dataset cifar100 --arch res2next29_6cx24wx6scale_se
```

### ImageNet
- Res2Net-50:
```
python train.py --dataset imagenet --imagenet-dir </path/to/data> --arch res2net50 --epoch 100 --milestones 30 --weight-decay 1e-4
```
- Res2Net-101:
```
python train.py --dataset imagenet --imagenet-dir </path/to/data> --arch res2net101 --epoch 100 --milestones 30 --weight-decay 1e-4
```
- Res2Net-152:
```
python train.py --dataset imagenet --imagenet-dir </path/to/data> --arch res2net152 --epoch 100 --milestones 30 --weight-decay 1e-4
```
- Res2NeXt-50_32x4d:
```
python train.py --dataset imagenet --imagenet-dir </path/to/data> --arch res2next50_32x4d --epoch 100 --milestones 30 --weight-decay 1e-4
```
- SE-Res2Net-50:
```
python train.py --dataset imagenet --imagenet-dir </path/to/data> --arch se_res2net50 --epoch 100 --milestones 30 --weight-decay 1e-4
```

## Results
Coming soon...

<!-- | Model                                           | Error rate |   Loss  | Error rate (paper) |
|:------------------------------------------------|:----------:|:-------:|:------------------:|
| WideResNet28-10 baseline                        |        3.82| 0.158   |                3.89|
| WideResNet28-10 +RICAP                          |    **2.82**| 0.141   |            **2.85**|
| WideResNet28-10 +Random Erasing                 |        3.18|**0.114**|                4.65|
| WideResNet28-10 +Mixup                          |        3.02| 0.158   |                3.02|

Learning curves of loss and accuracy.

![loss](loss.png)

![acc](acc.png) -->
