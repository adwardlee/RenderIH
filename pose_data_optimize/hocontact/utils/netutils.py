import torch


def rec_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def kmninit(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
