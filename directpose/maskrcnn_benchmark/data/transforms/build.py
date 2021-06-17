# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import pdb

def build_transforms_t(cfg, is_train=True):
    if True:#is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            )
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    transform = T.Compose(
        [
            #color_jitter,
            #T.Resize(min_size, max_size),
            #T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train else 0), #to get rid of randomness
            #T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform


def build_transforms(cfg, is_train=True,poop=False):
    if poop:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            )
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

        to_bgr255 = cfg.INPUT.TO_BGR255
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )
        color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        transform = T.Compose(
            [
                #color_jitter,
                #T.Resize(min_size, max_size),
                #T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train else 0), #to get rid of randomness
                #T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        return transform
    
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            )
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        print("min train",min_size)
        print("max train:", max_size)
        print("max test:", cfg.INPUT.MAX_SIZE_TEST)
        print("min test :", cfg.INPUT.MIN_SIZE_TEST)
        #pdb.set_trace()
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    print("crop size,,,", cfg.INPUT.CROP_SIZE)
    if is_train:
        crops = 600
        transform = T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomCrop(crops),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomCrop(0),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        
    return transform
