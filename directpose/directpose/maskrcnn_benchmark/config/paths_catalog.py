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
            # monkry data
        "monkey_train_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/monkey/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/monkey/annotations/monkey_train_v3.json",
            "kfun": "MonkeyKeypoints"
        },
        "monkey_val_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/monkey/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/monkey/annotations/monkey_val_v3.json",
            "kfun": "MonkeyKeypoints"
        },
        "monkey_test_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/monkey/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/monkey/annotations/monkey_train.json",
            "kfun": "MonkeyKeypoints"
        },
        "monkey_un_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/monkey/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/monkey/annotations/monkey_un_v3.json",
            "kfun": "MonkeyKeypoints"
        },
        #all fly (full for vid test)
         "fly_all_train_cocostyle": {
            "img_dir": "fly/train",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/fly/annotations/train_drosophila_annotations2020.json",
            "kfun": "FlyKeypoints"
        },
        "fly_vid_un_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/fly/unlabeled",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/fly/annotations/fly_vid_un.json",
            "kfun": "FlyKeypoints"
        },
        #fly data
        "fly_train_cocostyle": {
            "img_dir": "fly/train",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/fly/annotations/fly_train_v2.json",
            "kfun": "FlyKeypoints"
        },
        "fly_val_cocostyle": {
            "img_dir": "fly/train",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/fly/annotations/fly_val_v2.json",
            "kfun": "FlyKeypoints"
        },
        "fly_test_cocostyle": {
               "img_dir": "fly/train",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/fly/annotations/fly_train_v2.json",
            "kfun": "FlyKeypoints"
        },
        "fly_un_cocostyle": {
            "img_dir": "fly/train",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/fly/annotations/fly_un.json",
            "kfun": "FlyKeypoints"
        },
        # pup data
        "pup_train_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/pup_train_vf.json",
            "kfun": "PupKeypoints"
        },
        "pup_val_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/pup_val_area.json",
            "kfun": "PupKeypoints"
        },
        "pup_test_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/pup_train_vf.json",
            "kfun": "PupKeypoints"
        },
        #pup data padded
        "pup_padded_train_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/pup_train_padded.json",
            "kfun": "PupKeypoints"
        }, 
        #pup data only 3 kps
        "pup_full_train_cocostyle": {
            "img_dir": "pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/pups_with_3_kps.json",
            "kfun": "PupKeypoints"
        }, 
        "pup_full_val_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/pup_val_3_area.json",
            "kfun": "PupKeypoints"
        }, 
        #fake pup data only 3 kps
        "pup_fake_train_cocostyle": {
            "img_dir": "pup_fake/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup_fake/annotations/pups_fake_data.json",
            "kfun": "PupKeypoints"
        }, 
        #fake pup data only 3 kps v2
        "pup_fake_v2_train_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/pup_fake_v2/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup_fake_v2/annotations/pups_fake_data.json",
            "kfun": "PupKeypoints"
        }, 
        #padded pup val
        "pup_val_v2_cocostyle": {
            "img_dir": "/home/bsb2144/directpose/tools/datasets/pup/images",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup_fake_v2/annotations/pups_pad_val.json",
            "kfun": "PupKeypoints"
        }, 
        # unlabeled pup data
        "pup_train_cocostyle_un": {
            "img_dir": "pup/images/unlabeled/",
            "ann_file": "/home/bsb2144/directpose/tools/datasets/pup/annotations/train_pup_unan.json",
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
            "ann_file": "/home/bsb2144/directpose/tools/datasets/bee/annotations/validation-Copy1.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle": {
            "img_dir": "bee/train",
            "ann_file": "bee/annotations/train_bee_annotations2018_nondup.json",
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
        # bee resized 5 v2
        "bee_train_cocostyle_small_5_v2": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_5_v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 5 v2
        "bee_train_cocostyle_small_un_5v2": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 5 v3
        "bee_train_cocostyle_small_5_v3": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_5_v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 5 v3
        "bee_train_cocostyle_small_un_5v3": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 5 v4
        "bee_train_cocostyle_small_5_v4": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_5_v4.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 5 v4
        "bee_train_cocostyle_small_un_5v4": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v4.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 5 v5
        "bee_train_cocostyle_small_5_v5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_5_v5.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 5 v5
        "bee_train_cocostyle_small_un_5v5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v5.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 5 v5
        "bee_train_cocostyle_small_un_5v5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_5v5.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 25 v1
        "bee_train_cocostyle_small_25_v1": {
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
        # bee resized 25 v2
        "bee_train_cocostyle_small_25_v2": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_25_v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 25 v2
        "bee_train_cocostyle_small_un_25v2": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_25v2.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 25 v3
        "bee_train_cocostyle_small_25_v3": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_25_v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 25 v3
        "bee_train_cocostyle_small_un_25v3": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_25v3.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 25 v4
        "bee_train_cocostyle_small_25_v4": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_25_v4.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 25 v4
        "bee_train_cocostyle_small_un_25v4": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_25v4.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized 25 v5
        "bee_train_cocostyle_small_25_v5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_25_v5.json",
            "kfun": "BeeKeypoints"
        },
        # bee resized non labeled for 25 v5
        "bee_train_cocostyle_small_un_25v5": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_25v5.json",
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
        # bee resized unalbeled all
        "bee_train_cocostyle_small_un_all": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_all.json",
            "kfun": "BeeKeypoints"
        },
        "bee_val_cocostyle_small_un_all": {
            "img_dir": "bee_rescale/validation",
            "ann_file": "bee_rescale/annotations/validation.json",
            "kfun": "BeeKeypoints"
        },
        "bee_test_cocostyle_small_un_all": {
            "img_dir": "bee_rescale/train",
            "ann_file": "bee_rescale/annotations/train_bee_annotations2018_unan_for_all.json",
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
