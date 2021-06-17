# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import pdb
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator
from .transforms import build_transforms
#from .transforms import build_transforms_n


def build_dataset_n(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        print("data? :", data)
        factory = getattr(D, data["factory"])
        print("fact: ", factory)
        print("D: ", D)
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        print("args: ", args)
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    img_map = dataset.get_map()
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset], img_map

def check_conditions(indices):
    if (indices[0] not in range(0,135)) and (indices[0] not in range(216,351)):
        return False
    if (indices[429] not in range(0,135)) and (indices[429] not in range(216,351)):
        return False
    for i in range(0,142):
        if (indices[(i*3)] in range(0,135)) or (indices[(i*3)] in range(216,351)):
            #print("ho 1", indices[(i*3)])
            continue
        if (indices[((i*3)+1)] in range(0,135)) or (indices[(i*3)+1] in range(216,351)):
            #print("ho 2", indices[((i*3)+1)])
            continue
        if (indices[((i*3)+2)] in range(0,135)) or (indices[(i*3)+2] in range(216,351)):
            #print("ho 3", indices[((i*3)+2)])
            continue
        else:
            return False
            #print("baa")
            #print(indices[(i*3)], indices[((i*3)+1)], indices[((i*3)+2)])
    print("IND in checking: ", indices)
    return True

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        print("data yo")
        print(dataset)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        
#         train_sampleweights = []
#         print("data len", len(dataset))
#         for i in range(0, len(dataset)):
#             if i in range(0,135) or i in range(216, 351):
#                 train_sampleweights.append(0.00351)
#             else:
#                 train_sampleweights.append(0.0003125)      
        #sampler = WeightedRandomSampler(weights=train_sampleweights, num_samples = len(train_sampleweights))
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    print("DATASET", dataset, len(dataset))
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    #print("ag: ", aspect_grouping)
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        #print("group ids: ", group_ids)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
        #print("bs 1: ", batch_sampler)
        #for b in batch_sampler:
         #   print(b)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    #print("bs 2: ", batch_sampler)
    #for b in batch_sampler:
     #       print(b)
    return batch_sampler


def make_data_loader_og(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    print("train san check:", is_train)
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        #print("im p gpu: ", images_per_gpu)
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    print("DATA Path; ", dataset_list)

    transforms = build_transforms(cfg,is_train) #to test combo losses
    datasets, img_map = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []
    img_maps = []
    for dataset in datasets:
        #img_map = dataset.get_map()
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator
        )
        data_loaders.append(data_loader)
        img_maps.append(img_map)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return (data_loaders[0], img_maps[0])
    return data_loaders


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0,use_25=True,poop=False):
    num_gpus = get_world_size()
    print("train san check:", is_train)
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        print("im p gpu: ", images_per_gpu)
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    dataset_list_labeled = ('pup_train_cocostyle',)
    dataset_list_unlabeled = ('pup_train_cocostyle_un',)
    #dataset_list_unlabeled = ('bee_train_cocostyle_small_25_v1',)
    #pup_train_cocostyle
    #dataset_list_labeled = ('pup_train_cocostyle',)
    #dataset_list_unlabeled = ('bee_train_cocostyle_small_un_25v1',)
    #dataset_list_unlabeled = ('pup_train_cocostyle',)
    if poop == True:
        DatasetCatalog = paths_catalog.DatasetCatalog
        dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
        dataset_list = ('pup_val_cocostyle',)#('bee_val_cocostyle',)
        print("DATA Path; ", dataset_list)

        transforms = build_transforms(cfg,True,poop=True) #to test combo losses
        datasets, img_map = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

        data_loaders = []
        img_maps = []
        for dataset in datasets:
            return dataset

    transforms = build_transforms(cfg, is_train)
    
    datasets_labeled = build_dataset(dataset_list_labeled, transforms, DatasetCatalog, is_train)
    datasets_unlabeled = build_dataset(dataset_list_unlabeled, transforms, DatasetCatalog, is_train)

    print("DatasetCatalog", DatasetCatalog)
    print("dataset_l", datasets_labeled, len(datasets_labeled), len(datasets_labeled[0]), len(datasets_labeled[0][0]))
    
    data_loaders = []
    for (dataset_l, dataset_ul) in zip(datasets_labeled, datasets_unlabeled):
        print("len data l", len(dataset_l))
        #if use_25 == True:
         #   images_per_gpu_l = 25
        #else:
        images_per_gpu_l = 6
        sampler_l = make_data_sampler(dataset_l[0], shuffle, is_distributed)
        print("samp 1", list(sampler_l))
        batch_sampler_l = make_batch_data_sampler(
            dataset_l[0], sampler_l, aspect_grouping, images_per_gpu_l, num_iters, start_iter
        )
        collator_l = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers_l = cfg.DATALOADER.NUM_WORKERS
        
        images_per_gpu_ul = 3

        sampler_ul = make_data_sampler(dataset_ul[0], shuffle, is_distributed)
        print("samp 2", sampler_ul)
        batch_sampler_ul = make_batch_data_sampler(
            dataset_ul[0], sampler_ul, aspect_grouping, images_per_gpu_ul, num_iters, start_iter
        )
        collator_ul = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers_ul = cfg.DATALOADER.NUM_WORKERS
        
        
        data_loader_labeled = torch.utils.data.DataLoader(
            dataset_l[0],
            num_workers=num_workers_l,
            batch_sampler=batch_sampler_l,
            collate_fn=collator_l,
        )
        
        #pdb.set_trace()
        data_loader_unlabeled = torch.utils.data.DataLoader(
            dataset_ul[0],
            num_workers=num_workers_ul,
            batch_sampler=batch_sampler_ul,
            collate_fn=collator_ul,
        )
        
        print("lem lab" , len(data_loader_labeled))
        print("lem un" , len(data_loader_unlabeled))
        return (data_loader_labeled, data_loader_unlabeled)
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders



def make_data_loader_t(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    print("train san check:", is_train)
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        #print("im p gpu: ", images_per_gpu)
        shuffle = False
        num_iters = cfg.SOLVER.MAX_ITER
    

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    dataset_list = ('pup_val_cocostyle',)#('bee_val_cocostyle',)
    print("DATA Path; ", dataset_list)

    transforms = build_transforms(cfg,True) #to test combo losses
    datasets, img_map = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []
    img_maps = []
    for dataset in datasets:
        return dataset
        #img_map = dataset.get_map()
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator
        )
        data_loaders.append(data_loader)
        img_maps.append(img_map)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return (data_loaders[0], img_maps[0], datasets[0])
    return data_loaders

