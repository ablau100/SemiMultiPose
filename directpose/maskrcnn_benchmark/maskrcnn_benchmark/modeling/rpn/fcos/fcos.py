import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale
from maskrcnn_benchmark.layers import DFConv2d, DeformConv, \
    ModulatedDeformConv, DeformConvExtraStride
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints, BeeKeypoints, FlyKeypoints, PupKeypoints, MonkeyKeypoints


def biexp(x):
    return torch.where(x >= 0, torch.exp(x) - 1, 1 - torch.exp(-x))


def conv3x3(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        groups=groups
    )


def pad_to_target_size(features, target_h, target_w):
    N, C, H, W = features.size()
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    return F.pad(features, [0, pad_w, 0, pad_h])


class KPSPredictor(nn.Module):
    def __init__(self, in_channels, num_kps):
        super(KPSPredictor, self).__init__()
        self.kps_offsets = DeformConvExtraStride(
            in_channels, 2 * num_kps,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(
            self, kps_bases,
            sampled_features,
            sampled_feature_stride,
            fpn_stride
    ):
        assert kps_bases.size(1) == 2
        assert fpn_stride % sampled_feature_stride == 0
        stride = int(fpn_stride / sampled_feature_stride)
        padded_sampled_features = pad_to_target_size(
            sampled_features,
            stride * (kps_bases.size(2) - 1) + 1,
            stride * (kps_bases.size(3) - 1) + 1
        )
        offsets = self.kps_offsets(
            padded_sampled_features,
            kps_bases[:, [1, 0]].contiguous() * stride + stride // 2,
            stride
        )
        return offsets


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.cfg = cfg
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        
        if cfg.DATATYPE=='person':
            self.num_kps = 17
            self.group_to_kp_names = [
                ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
                ['left_shoulder'],
                ['left_elbow', 'left_wrist'],
                ['right_shoulder'],
                ['right_elbow', 'right_wrist'],
                ['left_hip'],
                ['left_knee', 'left_ankle'],
                ['right_hip'],
                ['right_knee', 'right_ankle']
            ]
        elif cfg.DATATYPE=='monkey':
            self.num_kps = 17
            self.group_to_kp_names = [
                ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
                ['left_shoulder'],
                ['left_elbow', 'left_wrist'],
                ['right_shoulder'],
                ['right_elbow', 'right_wrist'],
                ['left_hip'],
                ['left_knee', 'left_ankle'],
                ['right_hip'],
                ['right_knee', 'right_ankle']
            ]
        elif cfg.DATATYPE=='bee':
            self.num_kps = 5
            self.group_to_kp_names = [
                ['Tail', 'Head', 'Thorax', 'Rant', 'Lant']
            ]
        elif cfg.DATATYPE=='pup':
            self.num_kps = 3
            self.group_to_kp_names = [
                ['Head', 'Torso', 'Butt']
            ]
        elif cfg.DATATYPE=='fly':
            self.num_kps = 13
            self.group_to_kp_names = [
                ['head'],
                ['thorax'], 
                ['abdomen'], 
                ['wingL'], 
                ['wingR'], 
                ['forelegL4'], 
                ['forelegR4'], 
                ['midlegL4'], 
                ['midlegR4'], 
                ['hindlegL4'], 
                ['hindlegR4'], 
                ['eyeL'], 
                ['eyeR']
            ]
#             self.group_to_kp_names = [
#                 ['head', 'thorax', 'abdomen', 'wingL', 'wingR', 'forelegL4', 'forelegR4', 'midlegL4', 'midlegR4', 'hindlegL4', 'hindlegR4', 'eyeL', 'eyeR']
#             ]

        cls_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            # bbox_tower.append(
            #     nn.Conv2d(
            #         in_channels,
            #         in_channels,
            #         kernel_size=3,
            #         stride=1,
            #         padding=1
            #     )
            # )
            # bbox_tower.append(nn.GroupNorm(32, in_channels))
            # bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        # self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        num_kp_groups = len(self.group_to_kp_names)
        kps_tower_channels = 64 * num_kp_groups
        num_groups_gn = kps_tower_channels // 4

        kps_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            kps_tower.append(conv3x3(
                in_channels if i == 0 else kps_tower_channels,
                kps_tower_channels,
                groups=(1 if i == 0 else num_kp_groups)
            ))
            kps_tower.append(nn.GroupNorm(num_groups_gn, kps_tower_channels))
            kps_tower.append(nn.ReLU())
        self.kps_tower = nn.Sequential(*kps_tower)

        self.kps_bases = conv3x3(
            kps_tower_channels,
            2 * num_kp_groups,
            groups=num_kp_groups
        )

        self.sample_features_conv = nn.Sequential(
            conv3x3(kps_tower_channels, kps_tower_channels, groups=num_kp_groups),
            nn.GroupNorm(num_groups_gn, kps_tower_channels),
            nn.ReLU()
        )

        self.kps_offsets = nn.ModuleList()
        for g in self.group_to_kp_names:
            self.kps_offsets.append(KPSPredictor(
                kps_tower_channels // num_kp_groups, len(g)
            ))

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        # self.bbox_pred = nn.Conv2d(
        #     in_channels, 4, kernel_size=3, stride=1,
        #     padding=1
        # )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # self.kps_offsets2 = KPSPredictor(128, self.num_kps)

        # initialization
        for modules in [self.cls_tower, self.cls_logits,
                        self.centerness, self.kps_tower,
                        self.kps_bases, self.kps_offsets,
                        self.sample_features_conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d) or \
                        isinstance(l, DeformConv):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # self.scales = nn.ModuleList([
        #     Scale(init_value=1.0) for _ in range(len(self.fpn_strides))
        # ])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        all_kps_pred = []
        all_sampled_features = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            # bbox_reg.append(torch.exp(self.scales[l](
            #     self.bbox_pred(self.bbox_tower(feature))
            # )))

            kps_tower = self.kps_tower(feature)
            kps_bases = self.kps_bases(kps_tower)
            sampled_features = self.sample_features_conv(kps_tower)
            sampled_features = sampled_features.reshape(
                sampled_features.size(0),
                len(self.group_to_kp_names),
                -1,
                sampled_features.size(2),
                sampled_features.size(3),
            )
            all_sampled_features.append(sampled_features)

            all_group_kps_pred = []
            for i_group in range(len(self.group_to_kp_names)):
                per_group_kps_bases = kps_bases[:, 2 * i_group:2 * i_group + 2]
                if i_group == 0:  # if it's head group
                    per_group_kps_offsets = self.kps_offsets[i_group](
                        per_group_kps_bases, all_sampled_features[0][:, i_group],
                        self.fpn_strides[0], self.fpn_strides[l]
                    ) / float((self.fpn_strides[l] / self.fpn_strides[0]))
                else:
                    if l > 0:
                        sampled_features = all_sampled_features[l - 1][:, i_group]
                        sampled_features_stride = self.fpn_strides[l - 1]
                    else:
                        sampled_features = all_sampled_features[l][:, i_group]
                        sampled_features_stride = self.fpn_strides[l]

                    per_group_kps_offsets = self.kps_offsets[i_group](
                        per_group_kps_bases, sampled_features,
                        sampled_features_stride, self.fpn_strides[l]
                    ) / float(self.fpn_strides[l] / sampled_features_stride)

                per_group_kps_offsets = per_group_kps_offsets.reshape(
                    per_group_kps_offsets.size(0),
                    per_group_kps_offsets.size(1) // 2,
                    2,
                    per_group_kps_offsets.size(2),
                    per_group_kps_offsets.size(3)
                )
                all_group_kps_pred.append(
                    per_group_kps_bases[:, None] +
                    per_group_kps_offsets
                )

            kps_pred = self.merge_groups(all_group_kps_pred)
            
            if self.training:
                all_kps_pred.append(kps_pred)
            else:
                all_kps_pred.append(kps_pred * self.fpn_strides[l])

        return logits, bbox_reg, centerness, all_kps_pred

    def merge_groups(self, all_group_kps_pred):
        def get_index(kp_name):
            for i, g in enumerate(self.group_to_kp_names):
                for j, n in enumerate(g):
                    if n == kp_name:
                        return i, j
            assert False

        results = []
        if self.cfg.DATATYPE=='person':
            for kp_name in PersonKeypoints.NAMES:
                i, j = get_index(kp_name)
                results.append(all_group_kps_pred[i][:, j])
        elif self.cfg.DATATYPE=='bee':
            for kp_name in BeeKeypoints.NAMES:
                i, j = get_index(kp_name)
                results.append(all_group_kps_pred[i][:, j])
        elif self.cfg.DATATYPE=='fly':
            for kp_name in FlyKeypoints.NAMES:
                i, j = get_index(kp_name)
                results.append(all_group_kps_pred[i][:, j])
        elif self.cfg.DATATYPE=='pup':
            for kp_name in PupKeypoints.NAMES:
                i, j = get_index(kp_name)
                results.append(all_group_kps_pred[i][:, j])
        elif self.cfg.DATATYPE=='monkey':
            for kp_name in MonkeyKeypoints.NAMES:
                i, j = get_index(kp_name)
                results.append(all_group_kps_pred[i][:, j])
        return torch.cat(results, dim=1)


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None, heatmaps_results=None, heatmaps=None, stride=None, labeled=True):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        features = features[0:3]
        box_cls, box_regression, centerness, kps_pred = self.head(features)
        locations = self.compute_locations(features)
        
        
       
        if self.training:
            return self._forward_train(
                locations,
                box_cls,
                box_regression, 
                centerness,
                kps_pred,
                targets,
                heatmaps_results,
                heatmaps,
                images.image_sizes,
                stride,
                labeled
            )
        else:
            return self._forward_test(
                locations,
                box_cls,
                box_regression,
                centerness,
                kps_pred,
                heatmaps_results,
                heatmaps,
                images.image_sizes,
                stride
            )

    def _forward_train(
            self, locations, box_cls,
            box_regression, centerness,
            kps_pred, targets, heatmaps_results,
            heatmaps, image_sizes, stride,
            labeled=True
    ):
        if labeled == True:
            
            loss_box_cls, loss_centerness, loss_kps = self.loss_evaluator(
                locations, box_cls, box_regression, centerness, kps_pred, targets
            )
            losses = {
                "loss_cls": loss_box_cls,
                "loss_centerness": loss_centerness,
                "loss_kps": loss_kps
            }

        boxes = self.box_selector_test(
            locations,
            box_cls,
            box_regression,
            centerness,
            kps_pred,
            heatmaps_results,
            heatmaps,
            image_sizes,
            stride
        )
        
        if labeled == True:
            return {}, losses
        else:
            return boxes

    def _forward_test(
            self, locations, box_cls,
            box_regression, centerness,
            kps_pred, heatmaps_results, heatmaps, image_sizes, stride
    ):
        boxes = self.box_selector_test(
            locations,
            box_cls,
            box_regression,
            centerness,
            kps_pred,
            heatmaps_results,
            heatmaps,
            image_sizes,
            stride
        )

        return boxes, box_cls

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
