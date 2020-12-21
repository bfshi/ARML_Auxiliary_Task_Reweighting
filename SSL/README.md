Based on the repo [realistic-ssl-evaluation-pytorch](https://github.com/perrying/realistic-ssl-evaluation-pytorch).

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

Default setting is CIFAR 4000 labels. If you try other datasets or settings, please check the options first by ```python build_dataset.py -h```.

Running experiments (CIFAR 4000 labels, using S4L as SSL algorithm, task reweighting with ARML)

```python
python train.py --alg S4L --validation 1000 -d cifar10 --aux_task_num 2 --datanum 4000 --reweight_alg arml
```

