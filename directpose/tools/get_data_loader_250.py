# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys
import pdb

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def is_pytorch_1_1_0_or_later():
    return [int(_) for _ in torch.__version__.split(".")[:3]] >= [1, 1, 0]


def train(cfg, local_rank, distributed, loss_type):
#     assert is_pytorch_1_1_0_or_later()
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    weight_p = 'training_dir/250/standard/fcos_kps_ms_training_R_50_FPN_1x_bee/model_final.pth'
    original = torch.load(weight_p)
    new_m = {"model": original["model"]}
    torch.save(new_m, 'training_dir/250/standard/fcos_kps_ms_training_R_50_FPN_1x_bee/model_final.pth')
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    
    return data_loader



num_gpus = 1#int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1
local_rank = 0

if distributed:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

config_file = "/home/bsb2144/directpose/configs/fcos/fcos_kps_ms_training_R_50_FPN_1x.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 2, \
                     'DATATYPE', 'bee', \
        'OUTPUT_DIR', 'training_dir/' + number_data + '/' + loss_type + '_from_standard/fcos_kps_ms_training_R_50_FPN_1x_bee', \
        'SOLVER.KPS_GRAD_MULT', 10.0, \
        'SOLVER.MAX_GRAD_NORM', 5.0, \
        'SOLVER.POWER', 1.0, \
        'INPUT.CROP_SIZE', 800, \
        'INPUT.MIN_SIZE_RANGE_TRAIN', '(480, 1600)', \
        'INPUT.MAX_SIZE_TRAIN', 2666, \
        'MODEL.WEIGHT', 'training_dir/250/standard/fcos_kps_ms_training_R_50_FPN_1x_bee/model_final.pth', \
        'MODEL.RPN.BATCH_SIZE_PER_IMAGE', 1,\
        'MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE', 1,\
        'SOLVER.IMS_PER_BATCH', 3, \
        'SOLVER.MAX_ITER', 180000, \
        'MODEL.HEATMAPS_LOSS_WEIGHT', 4.0, \
        'DATASETS.TRAIN', "('bee_train_cocostyle', 'bee_train_cocostyle')"
        ])

output_dir = cfg.OUTPUT_DIR
if output_dir:
    mkdir(output_dir)


model = train(cfg, local_rank, distributed, loss_type)
