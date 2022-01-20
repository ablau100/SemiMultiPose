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

def train(cfg, local_rank, distributed, loss_type, alpha, beta, use_unlabeled=False, use_checkpoint=True):
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
    
    if use_checkpoint:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, just_weights=True)
    else:
        extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    
    if use_unlabeled:
        labeled_img_per_gpu = 2
        unlabeled_img_per_gpu = 1
    else:
        labeled_img_per_gpu = 3
        unlabeled_img_per_gpu = 0
        
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        use_unlabeled=use_unlabeled,
        labeled_img_per_gpu=labeled_img_per_gpu,
        unlabeled_img_per_gpu=unlabeled_img_per_gpu
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        loss_type,
        alpha=alpha,
        beta=beta,
        use_unlabeled=use_unlabeled
    )

    return model

# Load params from bash script
out_dir = str(sys.argv[1])
data_type = str(sys.argv[2])
alpha = float(sys.argv[3])
beta = float(sys.argv[4])
max_iters = int(sys.argv[5])

out_dir = 'training_dir/' + out_dir

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1
local_rank = 0

if distributed:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

# naming convention for datasets (see config/paths_catalog.py to upload your annotation files)
data_train = "('"+ data_type + "_train_cocostyle',)"
data_un = "('"+ data_type + "_unlabeled_cocostyle',)"

config_file = "../configs/fcos/fcos_kps_ms_training_R_50_FPN_1x.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 2, \
                     'DATATYPE', data_type, \
        'OUTPUT_DIR', out_dir, \
        'SOLVER.KPS_GRAD_MULT', 10.0, \
        'SOLVER.MAX_GRAD_NORM', 5.0, \
        'SOLVER.POWER', 1.0, \
        'SOLVER.CHECKPOINT_PERIOD', 1000, \
        'INPUT.CROP_SIZE', 600, \
        'INPUT.MIN_SIZE_RANGE_TRAIN', '(480, 1600)', \
        'INPUT.MAX_SIZE_TRAIN', 2666, \
        'MODEL.WEIGHT', '', \
        'MODEL.RPN.BATCH_SIZE_PER_IMAGE', 1,\
        'MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE', 1,\
        'SOLVER.IMS_PER_BATCH', 3, \
        'MODEL.HEATMAPS_LOSS_WEIGHT', 4.0, \
        'DATASETS.TRAIN', data_train, \
        'DATASETS.TEST', data_train, \
        'DATASETS.UNLABELED', data_un
        ])

output_dir = cfg.OUTPUT_DIR
if output_dir:
    mkdir(output_dir)

logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))


logger.info("Collecting env info (might take some time)")
logger.info("\n" + collect_env_info())

logger.info("Loaded configuration file {}".format(config_file))
with open(config_file, "r") as cf:
    config_str = "\n" + cf.read()
    logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))
 
    
# run model with only standard loss 
model_weight_part1 = ''
max_iter_part1 = round(int(max_iters * .2), -3) 

cfg.merge_from_list(['MODEL.WEIGHT', model_weight_part1, \
        'SOLVER.MAX_ITER', max_iter_part1, \
        ])

model_part1 = train(cfg, local_rank, distributed, 'standard', alpha, beta, False, False)

# run model with incorperating fusion loss on labeled frames
load_part1 = (7 - len(str(max_iter_part1))) * '0' + str(max_iter_part1)
model_weight_part2 = out_dir + '/model_' + load_part1 + '.pth'
max_iter_part2 = max_iter_part1 * 2

cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 2, \
        'MODEL.WEIGHT', model_weight_part2, \
        'SOLVER.MAX_ITER', max_iter_part2, \
        ])
model_part2 = train(cfg, local_rank, distributed, 'combined', alpha, beta, False, True)

# run model with incorperating fusion loss on unlabeled frames
load_part2 = (7 - len(str(max_iter_part2))) * '0' + str(max_iter_part2)
model_weight_part3 = out_dir + '/model_' + load_part2 + '.pth'
max_iter_part3 = max_iters

cfg.merge_from_list(['DATALOADER.NUM_WORKERS', 2, \
        'MODEL.WEIGHT', model_weight_part3, \
        'SOLVER.MAX_ITER', max_iter_part3, \
        ])
model_part3 = train(cfg, local_rank, distributed, 'combined', alpha, beta, True, True)






