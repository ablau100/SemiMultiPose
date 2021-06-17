# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, \
    WarmupPolyLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        #print("KEY VAL", key)
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        #print("weight_decay pre 1", weight_decay)
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            #print("weight_decay bias 2", weight_decay)

        if key.startswith("rpn.head") or key.startswith("heatmaps"):
            print("in starts with")
            if not key.endswith("scale"):
                print("apply SOLVER.KPS_GRAD_MULT to {}".format(key))
                lr *= cfg.SOLVER.KPS_GRAD_MULT
            else:
                print("do not apply SOLVER.KPS_GRAD_MULT to {}".format(key))
        
        #print("LRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRL",lr)
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    print("IN LR SCHEDULER")
    return WarmupPolyLR(
        optimizer,
        cfg.SOLVER.MAX_ITER,
        power=cfg.SOLVER.POWER,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
