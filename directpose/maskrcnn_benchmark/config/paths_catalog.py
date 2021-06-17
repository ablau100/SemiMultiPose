# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        # pp og
        "pp_train_cocostyle": {
            "img_dir": "coco/images",
            "ann_file": "coco/annotations/person_keypoints_train_1564.json",
            "kfun": "PersonKeypoints"
        },
        "pp_val_cocostyle": {
            "img_dir": "coco/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/coco/annotations/person_keypoints_val_782.json",
            "kfun": "PersonKeypoints"
        },
        "pp_test_cocostyle": {
            "img_dir": "coco/images",
            "ann_file": "coco/annotations/person_keypoints_train_1564.json",
            "kfun": "PersonKeypoints"
        },
        # pp un
        "pp_train_cocostyle_un": {
            "img_dir": "coco/images",
            "ann_file": "coco/annotations/person_keypoints_train_unlabeled_909.json",
            "kfun": "PersonKeypoints"
        },
        "ppu_val_cocostyle": {
            "img_dir": "coco/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/coco/annotations/person_keypoints_val_782.json",
            "kfun": "PersonKeypoints"
        },
        "ppu_test_cocostyle": {
            "img_dir": "coco/images",
            "ann_file": "coco/annotations/person_keypoints_train_unlabeled_909.json",
            "kfun": "PersonKeypoints"
        },
        
        # pup og
        "pup_train_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "pup/pup_train2.json",
            "kfun": "PupKeypoints"
        },
        "pup_val_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/pup_val2.json",
            "kfun": "PupKeypoints"
        },
        "pup_test_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "pup/pup_train2.json",
            "kfun": "PupKeypoints"
        },
        # pup unlabeled
        "pup_train_cocostyle_un": {
            "img_dir": "pup/images/unlabeled",
            "ann_file": "pup/annotations/train_pup_unan.json",
            "kfun": "PupKeypoints"
        },
        "pup_val_cocostyle_un": {
            "img_dir": "pup/images",
            "ann_file": "pup/pup_val2.json",
            "kfun": "PupKeypoints"
        },
        "pup_test_cocostyle_un": {
            "img_dir": "pup/images/unlabeled",
            "ann_file": "pup/annotations/train_pup_unan.json",
            "kfun": "PupKeypoints"
        },
        # bee
        "bee_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_nondup.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_nondup.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized
        "bee_train_cocostyle_small": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_nondup.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_nondup.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 25 v1
        "bee_train_cocostyle_small_25_v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_25_v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_25_v1": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_25_v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_25_v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 25 v1
        "bee_train_cocostyle_small_un_25v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_25v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_un_25v1": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_un_25v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_25v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 10 v1
        "bee_train_cocostyle_small_10_v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_10_v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_10_v1": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_10_v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_10_v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 10 v1
        "bee_train_cocostyle_small_un_10v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_10v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_un_10v1": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_un_10v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_10v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 5 v1
        "bee_train_cocostyle_small_5_v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_5_v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_5_v1": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_5_v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_5_v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 5 v1
        "bee_train_cocostyle_small_un_5v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_un_5v1": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_un_5v1": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized unalbeled all
        "bee_train_cocostyle_small_un_all": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan2.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_un_all": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_un_all": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan2.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized unalbeled 5
        "bee_train_cocostyle_small_un_5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan5.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_un_5": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_un_5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan5.json",
            "kfun": "BeeKeypoints"
        },
        # bee unlabeld only
        "bee_train_cocostyleun": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_unn5im.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyleun": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyleun": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_unn5im.json",
            "kfun": "BeeKeypoints"
        },
        # bee unan
        "beeua_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_unan2.json",
            "kfun": "BeeKeypoints"
        },
        "beeua_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "beeua_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_unan2.json",
            "kfun": "BeeKeypoints"
        },
        #bee 10 high nc combo
        "bee_train_cocostyle_10": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_nw_3.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_10": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_10": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_nw_3.json",
            "kfun": "BeeKeypoints"
        },
        # bee 100 v0
        "bee_100_v0_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v0.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v0_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v0_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v0.json",
            "kfun": "BeeKeypoints"
        },
        # bee 100 v1
        "bee_100_v1_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v1_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v1_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee 100 v2
        "bee_100_v2_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v2.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v2_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v2_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee 100 v3
        "bee_100_v3_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v3.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v3_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_100_v3_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_100_v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee 50 v0
        "bee_50_v0_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v0.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v0_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v0_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v0.json",
            "kfun": "BeeKeypoints"
        },
        # bee 50 v1
        "bee_50_v1_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v1_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v1_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee 50 v2
        "bee_50_v2_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v2.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v2_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v2_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee 50 v3
        "bee_50_v3_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v3.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v3_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_v3_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_50_v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee 25 v0
        "bee_25_v0_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v0.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v0_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v0_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v0.json",
            "kfun": "BeeKeypoints"
        },
        # bee 25 v1
        "bee_25_v1_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v1_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v1_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v1.json",
            "kfun": "BeeKeypoints"
        },
        # bee 25 v2
        "bee_25_v2_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v2.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v2_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v2_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee 25 v3
        "bee_25_v3_train_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v3.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v3_val_cocostyle": {
            "img_dir": "bee/validation",
            "ann_file": "bee/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_v3_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_25_v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee 50
        "bee_50_train_cocostyle": {
            "img_dir": "bee_50/train",
            "ann_file": "bee_50/annotations/train_bee_annotations2018.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_val_cocostyle": {
            "img_dir": "bee_50/validation",
            "ann_file": "bee_50/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_50_test_cocostyle": {
            "img_dir": "bee_50/train",
            "ann_file": "bee_50/annotations/train_bee_annotations2018.json",
            "kfun": "BeeKeypoints"
        },
        # bee 25
        "bee_25_train_cocostyle": {
            "img_dir": "bee_25/train",
            "ann_file": "bee_25/annotations/train_bee_annotations2018.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_val_cocostyle": {
            "img_dir": "bee_25/validation",
            "ann_file": "bee_25/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_25_test_cocostyle": {
            "img_dir": "bee_25/train",
            "ann_file": "bee_25/annotations/train_bee_annotations2018.json",
            "kfun": "BeeKeypoints"
        },
        # drosophila
        "drosophila_train_cocostyle": {
            "img_dir": "drosophila/train",
            "ann_file": "drosophila/annotations/train_drosophila_annotations2020.json",
            "kfun": "FlyKeypoints"
        },
        "drosophila_val_cocostyle": {
            "img_dir": "drosophila/validation",
            "ann_file": "drosophila/annotations/validation_drosophila_annotations2020.json",
            "kfun": "FlyKeypoints"
        },
        "drosophila_test_cocostyle": {
            "img_dir": "drosophila/train",
            "ann_file": "drosophila/annotations/train_drosophila_annotations2020.json",
            "kfun": "FlyKeypoints"
        }
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if attrs.get('kfun') is not None:
                args = dict(
                    root=os.path.join(data_dir, attrs["img_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_file"]),
                    kfun=attrs["kfun"],
                )
            else:
                args = dict(
                    root=os.path.join(data_dir, attrs["img_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_file"]),
                    kfun="PersonKeypoints",
                )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
