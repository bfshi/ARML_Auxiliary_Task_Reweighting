# Requirements
- Python 3.6+
- PyTorch 1.1.0
- torchvision 0.3.0
- numpy 1.16.2

# How to run
Prepare dataset

```python
python build_dataset.py
```

Default setting is SVHN 1000 labels. If you try other settings, please check the options first by ```python build_dataset.py -h```.

Running experiments (SVHN 1000 labels as default)

```python
python train.py -a S4L --validation 2000 -d svhn --aux_task_num 2 --datanum 1000
```

Testing

```python
python test.py -a S4L --validation 2000 -d svhn --aux_task_num 2 --datanum 1000 --resume <PATH TO YOUR MODEL>
```

# Performance

|                   | CIFAR10 (4000 labels) | SVHN (1000 labels) |
| :---------------: | :-------------------: | :----------------: |
|   S4L (Uniform)   |    15.67 $\pm$ .29    |   7.83 $\pm$ .33   |
|   S4L + AdaLoss   |    21.06 $\pm$ .17    |  11.53 $\pm$ .39   |
|  S4L + GradNorm   |    14.07 $\pm$ .44    |   7.68 $\pm$ .13   |
|  S4L + CosineSim  |    15.03 $\pm$ .31    |   7.02 $\pm$ .25   |
|   S4L + OL_AUX    |    16.07 $\pm$ .51    |   7.82 $\pm$ .32   |
| S4L + GridSearch  |    13.76 $\pm$ .22    |   6.07 $\pm$ .17   |
| S4L + ARML (ours) |    13.68 $\pm$ .35    |   5.89 $\pm$ .22   |

