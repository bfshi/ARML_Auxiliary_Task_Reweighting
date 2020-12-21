import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from lib.utils import Momentum_Average, Variance_Estimator, cal_grad_norm

class Task_Weighter(nn.Module):
    def __init__(self, task_num):
        super(Task_Weighter, self).__init__()
        self.task_num = task_num
        self.alpha = torch.ones(task_num - 1, requires_grad=True)
        self.alpha = nn.Parameter(self.alpha)

        self.average_aux_loss = Momentum_Average()
        self.gradient_variance_estimator = Variance_Estimator()

    def forward(self, losses):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)
        return main_loss + (aux_loss * self.alpha).sum()

    def gradient_variance_update(self, model):
        for name, param in model.named_parameters():
            self.gradient_variance_estimator.update(param.grad.data)
            break

    def inject_grad_noise(self, model, noise_var, lr, momentum):
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data + \
                                  torch.randn_like(param.grad.data) * noise_var**0.5 * (1 - momentum)**0.5 / lr * (param.grad.data**2)**0.25

    def weight_loss(self, model, losses, reweight_alg):
        weight_loss = None
        if reweight_alg == 'arml':
            weight_loss = self.arml_weight_loss(model, losses)
        elif reweight_alg == 'gradnorm':
            weight_loss = self.gradnorm_weight_loss(model, losses)  # gradnorm
        elif reweight_alg == 'ol_aux':
            weight_loss = self.ol_aux_weight_loss(model, losses)  # ol_aux
        return weight_loss

    def arml_weight_loss(self, model, losses):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)

        loss_grad_gap = grad(main_loss - (aux_loss * self.alpha).sum(), model.parameters(),
                          create_graph=True, allow_unused=True)

        alpha_loss = sum([grd.norm()**2 for grd in loss_grad_gap[:-6]])
        return alpha_loss

    def ada_loss_forward(self, model, losses):
        main_loss = losses[0]
        losses[2] = losses[2] + 1
        aux_loss = torch.log(torch.stack(losses[1:], dim=0) + 1e-6)
        return main_loss + (aux_loss * self.alpha).sum()

    def gradnorm_weight_loss(self, model, losses):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)
        self.average_aux_loss.update(aux_loss.detach())
        r = aux_loss.detach() / self.average_aux_loss.init
        r = r / r.mean()

        alpha_loss = 0
        grad_norms = []
        for i in range(1, self.task_num):
            grad_i = grad(aux_loss[i - 1], model.unit3.parameters(),
                            create_graph=True, allow_unused=True)
            grad_norms.append(self.alpha[i - 1] * grad_i[-1].detach().norm())

        grad_norm_mean = sum(grad_norms) / (self.task_num - 1)
        for i in range(1, self.task_num):
            alpha_loss += torch.abs(grad_norms[i - 1] - grad_norm_mean * r[i - 1]**1.0)

        return alpha_loss

    def cosine_similarity_forward(self, model, losses, if_update_weight):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)

        if if_update_weight:
            main_grad = grad(main_loss, model.unit3.parameters(), retain_graph=True)[-1].detach()
            for i in range(1, self.task_num):
                grad_i = grad(aux_loss[i - 1], model.unit3.parameters(), retain_graph=(i < self.task_num - 1))[-1].detach()
                self.alpha.data[i - 1] = int((grad_i * main_grad).sum() > 0)

        return main_loss + (aux_loss * self.alpha).sum()

    def ol_aux_weight_loss(self, model, losses):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)

        main_loss_grads = grad(main_loss, model.parameters(),
                          create_graph=True, allow_unused=True)
        aux_loss_grads = grad((aux_loss * self.alpha).sum(), model.parameters(),
                          create_graph=True, allow_unused=True)

        alpha_loss = 0
        for main_grad, aux_grad in zip(main_loss_grads[:-6], aux_loss_grads[:-6]):
            alpha_loss += -(main_grad.detach() * aux_grad).sum()
        return alpha_loss



def task_weighter(task_num):
    return Task_Weighter(task_num)