# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
import numpy as np
import math
import h5py
import pdb
import pickle

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list


global_step = None


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loaders,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    loss_type,
    alpha=1,
    beta=1,
    version=0,
    use_unlabeled = False,
    tr = 'all'
):
    
    #data_loader = data_loaders[0]
    #img_map = data_loaders[1]
    
    checkpoint_period = 1000
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    #max_iter = len(data_loaders[0])
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    
    if use_unlabeled:
        dl_1 = data_loaders[0]
        dl_2 = data_loaders[1]
        max_iter = len(data_loaders[0])
        print("DL1: ", dl_1, len(dl_1), len(dl_1.dataset))
        print("DL2: ", dl_2, len(dl_2), len(dl_2.dataset))
    else:
        dl_1 = data_loaders
        dl_2 = data_loaders
        max_iter = len(data_loaders)
        print("DL1: ", dl_1, len(dl_1))
    
    for iteration, data in enumerate(zip(dl_1 , dl_2), start_iter):
        
        images_1 = data[0][0]
        images_2 = data[1][0]
        targets_1 = data[0][1]
        targets_2 = data[1][1]
        
        data_time = time.time() - end
        iteration = iteration + 1
        global global_step
        global_step = iteration
        arguments["iteration"] = iteration
        
        images_la = images_1.to(device)
        targets_la = [target.to(device) for target in targets_1]
        
        images_un = images_2.to(device)
        targets_un = [target.to(device) for target in targets_2]
        
        images = data[0][0]
        targets = data[0][1]
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        #print('MODEL', model)
        loss_dict, diff_dict = model(images, targets, images_la,
                                     targets_la,images_un,targets_un,
                                     loss_type, alpha, beta, use_unlabeled)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
 
        #send stuff to csv
        
        loss_eval2 = loss_dict_reduced.copy()
        diff_dict2 = diff_dict.copy()
        
        out_dir = "results/fp_25" + str(loss_type)+ '_v' + str(version) + '_iter' + str(iteration) +'_uu'+alpha+ str(use_unlabeled) + str(beta)+'.pkl'
        if iteration % 1000 == 0:
            with open(out_dir, 'wb') as f:
                pickle.dump(diff_dict2, f)
               
  
        
#         def firstNan(listfloats):
#             for i, item in enumerate(listfloats):
#                 if math.isnan(item):
#                     return i
        
#         log_error_rmse_file = "diff_file_" + str(loss_type) + str(version) + '_' + str(alpha)+ '_' + str(beta)+ '_' + str(use_unlabeled)+'.h5'
#         print("ITER", iteration)
#         if iteration == start_iter + 1:
#             print("IN MAKE FILE")
#             data_tmp = np.zeros(max_iter)*np.nan
#             with h5py.File(log_error_rmse_file, "w") as f:
#                 for k_loss_eval2, v_loss_eval2 in diff_dict2.items():
#                     print("making the diff h5", k_loss_eval2)
#                     f.create_dataset(k_loss_eval2, data=data_tmp)#, max_shape=(maxiters,))
#                     #dset = f.create_dataset('', )
                    
#         with h5py.File(log_error_rmse_file, "a") as f:
#             for k_loss_eval2, v_loss_eval2 in diff_dict2.items():
#                 print("k_loss_eval2",k_loss_eval2)
#                 print("v_loss_eval2",v_loss_eval2)
#                 print('before', f[k_loss_eval2][:30])
#                 start = firstNan(f[k_loss_eval2][:])
#                 end = start + len(v_loss_eval2)
#                 for idx, item in enumerate(v_loss_eval2):
#                     f[k_loss_eval2][start+idx] = v_loss_eval2#item
#                 #f[k_loss_eval2][start:end] = v_loss_eval2
#                 print('after', f[k_loss_eval2][:30])
#                 if end >= 10:
#                     s
                
                
                
#                 if isinstance(v_loss_eval2, (float)) or isinstance(v_loss_eval2, (int)):
#                     f[k_loss_eval2][(iteration-1)] =v_loss_eval2#.detach().cpu().numpy()
#                 else:
#                     f[k_loss_eval2][(iteration-1)] =v_loss_eval2.detach().cpu().numpy()
                
                
       
        
            
        optimizer.zero_grad()
        losses.backward()

        if cfg.SOLVER.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.SOLVER.MAX_GRAD_NORM
            )
        print("LR pre opt step", optimizer.param_groups[0]["lr"])
        optimizer.step()
        print("LR pre sched step", optimizer.param_groups[0]["lr"])
        scheduler.step()
        print("LR post steps", optimizer.param_groups[0]["lr"])

        #pdb.set_trace()
        
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = 'placeholder'#str(datetime.timedelta(seconds=int(eta_seconds)))
        #max_iter = 10000
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:#TO REMOVE
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
