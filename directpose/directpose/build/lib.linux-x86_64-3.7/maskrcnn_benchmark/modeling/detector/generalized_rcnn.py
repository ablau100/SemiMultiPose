# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..heatmaps.heatmaps import build_heatmaps
from maskrcnn_benchmark.config import cfg


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.heatmaps = build_heatmaps(cfg, [256] * 5)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        # import cv2
        # image = images.tensors[0].cpu().permute(1, 2, 0) + torch.tensor(cfg.INPUT.PIXEL_MEAN)
        # cv2.imshow("image", image.byte().cpu().numpy())

        body_features = self.backbone.body(images.tensors)
        im_hw = images.tensors.size(2), images.tensors.size(3)
        features = self.backbone.fpn(body_features)

        if self.training:
            heatmaps_results, heatmaps_losses, heatmaps = self.heatmaps(features, targets, im_hw=im_hw)
            stride = None
        else:
            heatmaps_results, stride, heatmaps = self.heatmaps(features, None, im_hw=im_hw)
        proposals, proposal_losses = self.rpn(images, features, targets, heatmaps_results, heatmaps, stride)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(heatmaps_losses)
            return losses

        return result
