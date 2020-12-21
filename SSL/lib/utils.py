class Momentum_Average():
    def __init__(self, momentum=0.9):
        self.avg = None
        self.init = None
        self.momentum = momentum

    def update(self, v):
        if self.avg is None:
            self.avg = v
            self.init = v
        else:
            self.avg = self.momentum * self.avg + (1 - self.momentum) * v


class Variance_Estimator():
    def __init__(self, sample_num = 10):
        self.samples = []
        self.sample_num = sample_num
        self.avg = None
        self.var = None
        self.mean_var = None

    def update(self, v):
        self.samples.append(v)
        if len(self.samples) > self.sample_num:
            del(self.samples[0])
        self.avg = sum(self.samples) / len(self.samples)
        self.var = sum([sample**2 for sample in self.samples]) / len(self.samples) - self.avg**2
        self.mean_var = self.var.mean()


def cal_grad_norm(model, loss):
    loss.backward(retain_graph=True)
    grad_norm = 0
    for name, param in model.named_parameters():
        if 'classifier' in name or param.grad is None:
            continue
        grad_norm += param.grad.data.detach().norm() ** 2
    model.zero_grad()

    return grad_norm ** 0.5