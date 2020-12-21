import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torchvision
from config import config
from lib.algs.vat import VAT

eps = 1e-6

class S4L(nn.Module):
    def __init__(self, model, ema_factor, dataset):
        super(S4L, self).__init__()
        self.random_rotation = MyRotationTransform()
        self.random_aug = RandomAug()
        self.exemplar_mt = Exemplar_MT(model, ema_factor)

    def rotate_loss(self, x, model):
        x_rotated = torch.cat([x, x, x, x], dim=0).clone()
        target = torch.zeros(x.shape[0] * 4, device=x.device, dtype=torch.long)
        bs = x.shape[0]
        for i in range(4):
            x_rotated[i * bs: (i + 1) * bs] = self.random_rotation(x, i)
            target[i * bs: (i + 1) * bs] = i
        output = model(x_rotated, mode='rotate')
        loss = F.cross_entropy(output, target)
        return loss

    def exemplar_mt_loss(self, x, feature, model):
        f_key, f_query = self.exemplar_mt(x, feature, model)
        loss = -(F.normalize(f_key, dim=1) * F.normalize(f_query, dim=1)).sum(dim=1).mean()
        return loss

    def moving_average(self, parameters):
        self.exemplar_mt.moving_average(parameters)

    def forward(self, x, feature, y, model, mask):
        rotate_loss = self.rotate_loss(x, model)
        exemplar_mt_loss = self.exemplar_mt_loss(x, feature, model)
        return torch.cat([rotate_loss.view(1), exemplar_mt_loss.view(1)])

class MyFlip(nn.Module):
    def __init__(self):
        super(MyFlip, self).__init__()

    def forward(self, x, if_flip):
        if if_flip:
            x = x.flip(-1)
        return x

class MyRotationTransform(nn.Module):
    """Rotate by one of the given angles (anti-clockwise)."""

    def __init__(self):
        super(MyRotationTransform, self).__init__()
        self.angles = [0, 90, 180, 270]

    def forward(self, x, angle_choice):
        angle = self.angles[angle_choice]
        if angle == 90:
            x = x.transpose(-2, -1).flip(-2)
        elif angle == 180:
            x = x.flip(-2).flip(-1)
        elif angle == 270:
            x = x.transpose(-2, -1).flip(-1)
        return x

class MyGaussianNoise(nn.Module):
    def __init__(self):
        super(MyGaussianNoise, self).__init__()
        self.scale = 0.03

    def forward(self, x):
        xmin = x.min()
        xmax = x.max()
        xmean = x.mean()
        noise = (torch.randn_like(x) * self.scale + xmean) * (xmax - xmin) / 2
        return x + noise

class MyCutOut(nn.Module):
    def __init__(self):
        super(MyCutOut, self).__init__()
        self.scale = 0.5

    def forward(self, x):
        h = x.shape[-2]
        w = x.shape[-1]
        hcut = int(h * self.scale)
        wcut = int(w * self.scale)
        center = (random.randrange(h), random.randrange(w))
        x[:, :, max(center[0] - hcut//2, 0):min(center[0] + hcut//2, h), max(center[1] - wcut//2, 0):min(center[1] + wcut//2, w)] = 0
        return x

class RandomAug(nn.Module):
    def __init__(self):
        super(RandomAug, self).__init__()
        self.rotate = MyRotationTransform()
        self.flip = MyFlip()
        self.gaussian_noise = MyGaussianNoise()
        self.cutout = MyCutOut()

    def forward(self, x):
        x = self.flip(x, random.randrange(2))
        x = self.gaussian_noise(x)
        x = self.cutout(x)
        return x

class Exemplar_MT(nn.Module):
    def __init__(self, model, ema_factor):
        super().__init__()
        self.model = model
        self.model.train()
        self.ema_factor = ema_factor
        self.global_step = 0
        self.random_aug = RandomAug()

    def forward(self, x, feature, model):
        self.global_step += 1
        _, f_hat = self.model(x, return_feature=True)
        model.update_batch_stats(False)
        _, f = model(self.random_aug(x), return_feature=True)
        model.update_batch_stats(True)
        return f_hat.detach(), f

    def mt_forward(self, x, feature, y, model, mask):
        self.global_step += 1
        y_hat = self.model(x)
        model.update_batch_stats(False)
        y = model(self.random_aug(x))
        model.update_batch_stats(True)
        return (F.mse_loss(y.softmax(1), y_hat.softmax(1).detach(), reduction="none").mean(1) * mask).mean()

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
        for emp_p, p in zip(self.model.parameters(), parameters):
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data