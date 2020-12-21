#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor

import argparse, math, time, json, os

from lib import wrn, transform
from lib.task_weighter import task_weighter
from config import config

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="supervised", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT, SSL]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=10000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="svhn", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument("--aux_task_num", default=1, type=int, help="how many auxiliary tasks to use (default 1)")
parser.add_argument("--datanum", default=4000, type=int, help="how much labeled data (default: 4000 for cifar10, 1000 for svhn)")
parser.add_argument("--resume", default="none", type=str, help="path to the model you want to test performance of")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

condition = {}
exp_name = ""

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)

l_train_dataset = dataset_cfg["dataset"](args.root, "l_train", labeled_num=args.datanum)
u_train_dataset = dataset_cfg["dataset"](args.root, "u_train", labeled_num=args.datanum)
val_dataset = dataset_cfg["dataset"](args.root, "val")
test_dataset = dataset_cfg["dataset"](args.root, "test")

print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {
    "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
    "validation":len(val_dataset), "test":len(test_dataset)
}

class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

shared_cfg = config["shared"]
if args.alg != "supervised":
    # batch size = 0.5 x batch size
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
    )
else:
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
    )
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
    u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
)

val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

if args.em > 0:
    print("entropy minimization : {}".format(args.em))
    exp_name += "em_"
condition["entropy_maximization"] = args.em

model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])
# optimizer = optim.SGD(model.parameters(), lr=alg_cfg["lr"], momentum=0.9)

task_weighter = task_weighter(args.aux_task_num + 1).to(device)
optimizer_task_weighter = optim.SGD(task_weighter.parameters(), lr=shared_cfg["task_weighter_lr"])

model.load_state_dict(torch.load(args.resume))

trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT
    ssl_obj = VAT(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT
    t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI": # PI Model
    from lib.algs.pimodel import PiModel
    ssl_obj = PiModel()
elif args.alg == "ICT": # interpolation consistency training
    from lib.algs.ict import ICT
    t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
elif args.alg == "MM": # MixMatch
    from lib.algs.mixmatch import MixMatch
    ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
elif args.alg == 'SSL': #SSL
    from lib.algs.s4l import S4L
    t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = S4L(t_model, alg_cfg["ema_factor"])
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))

print()
iteration = 0
maximum_val_acc = 0
maximum_test_acc = 0
s = time.time()

with torch.no_grad():
    model.eval()
    print()
    print("### validation ###")
    sum_acc = 0.
    s = time.time()
    for j, data in enumerate(val_loader):
        input, target = data
        input, target = input.to(device).float(), target.to(device).long()

        output = model(input)

        pred_label = output.max(1)[1]
        sum_acc += (pred_label == target).float().sum()
        if ((j + 1) % 10) == 0:
            d_p_s = 10 / (time.time() - s)
            print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                j + 1, len(val_loader), d_p_s, (len(val_loader) - j - 1) / d_p_s
            ), "\r", end="")
            s = time.time()
    acc = sum_acc / float(len(val_dataset))
    print()
    print("varidation accuracy : {}".format(acc))
    # test
    # if maximum_val_acc < acc:
    if True:
        print("### test ###")
        maximum_val_acc = acc
        sum_acc = 0.
        s = time.time()
        for j, data in enumerate(test_loader):
            input, target = data
            input, target = input.to(device).float(), target.to(device).long()
            output = model(input)
            pred_label = output.max(1)[1]
            sum_acc += (pred_label == target).float().sum()
            if ((j + 1) % 10) == 0:
                d_p_s = 100 / (time.time() - s)
                print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
                    j + 1, len(test_loader), d_p_s, (len(test_loader) - j - 1) / d_p_s
                ), "\r", end="")
                s = time.time()
        print()
        test_acc = sum_acc / float(len(test_dataset))
        maximum_test_acc = max(maximum_test_acc, test_acc)
        print("test accuracy : {}".format(test_acc))
        # torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))
model.train()
s = time.time()

print("best test acc : {}".format(maximum_test_acc))
condition["test_acc"] = test_acc.item()

exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
    json.dump(condition, f)
