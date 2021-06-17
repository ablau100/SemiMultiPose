# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch, gc
#import torch.cuda.memory_summary
from torch import nn
import pdb

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..heatmaps.heatmaps import build_heatmaps
from ..mse.mse import get_mse_per_batch
from ..mse.mse import get_tps
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
        #self.mse = _mse()
        
        
    def forward(self, images, targets=None, images_la=None,
                targets_la=None, images_un=None,
                targets_un=None,loss_type=None, alpha=1,
                beta=1, has_unlabeled = False):
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
        

        #body_features = self.backbone.body(images.tensors)
        im_hw = images.tensors.size(2), images.tensors.size(3)
        #features = self.backbone.fpn(body_features)
        
        if self.training:
            if has_unlabeled == True:
                body_features_la = self.backbone.body(images_la.tensors)
                features_la = self.backbone.fpn(body_features_la)
                body_features_un = self.backbone.body(images_un.tensors)
                features_un = self.backbone.fpn(body_features_un)
            else:
                body_features = self.backbone.body(images.tensors)
                features = self.backbone.fpn(body_features)
        
        if self.training:
            if has_unlabeled == False:
                heatmaps_results_all, heatmaps_losses, heatmaps_all = self.heatmaps(features, 
                                                                        targets, im_hw=im_hw, 
                                                                        ground_truth=True,
                                                                          labeled = True)
                stride = im_hw[0] // heatmaps_all.size(2)
            else:
                heatmaps_results_un, heatmaps_un = self.heatmaps(features_un, 
                                                                           targets_un, im_hw=im_hw, 
                                                                            ground_truth=True,
                                                                              labeled = False)
                heatmaps_results_la, heatmaps_losses, heatmaps_la = self.heatmaps(features_la, 
                                                                            targets_la, im_hw=im_hw, 
                                                                            ground_truth=True,
                                                                           labeled = True)
                stride = im_hw[0] // heatmaps_la.size(2)
        else:
            body_features = self.backbone.body(images.tensors)
            im_hw = images.tensors.size(2), images.tensors.size(3)
            features = self.backbone.fpn(body_features)
            heatmaps_results_all, stride, heatmaps_all, c_loss = self.heatmaps(features, targets=None, im_hw=im_hw)
            
        if self.training:
            if has_unlabeled == True:
                proposals_la, proposal_losses = self.rpn(images_la, 
                                                         features_la, 
                                                         targets_la,
                                                         heatmaps_results_la, 
                                                         heatmaps_la, stride,
                                                         labeled=True)
                proposals_un = self.rpn(images_un, 
                                        features_un, 
                                        targets_un,
                                        heatmaps_results_un, 
                                        heatmaps_un, stride,
                                        labeled=False)
            else:  
                proposals_all, proposal_losses = self.rpn(images, features, targets, heatmaps_results_all, heatmaps_all, stride,labeled=True)
        
            
        
        if not self.training:
            proposals_all = self.rpn(images, features, targets, heatmaps_results_all, heatmaps_all, stride,labeled=False)
        
        #print("prop s", proposals_all[0].get_field("scores"))
       # print("props grcnn")
        #print(proposals_all)
        #print("0time", proposals_all[0])

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            if self.training:
                # RPN-only models don't have roi_heads
                #x = features
                if has_unlabeled == True:
                    result = proposals_la
                else:
                    result = proposals_all
                detector_losses = {}
        if not self.training: 
            result = proposals_all 
        #print("result", result[0])
        #print("target",targets)

        
       
        if loss_type == "combined" or loss_type == "combined_og":
            if has_unlabeled == True:
                heatmaps_results_combined, combined_losses, heatmaps_combined = self.heatmaps(features_la, proposals_la, im_hw=im_hw, ground_truth=False, com=True,alpha=float(alpha),un=False)

                heatmaps_results_combined_un, unlabeled_losses, heatmaps_combined_un = self.heatmaps(features_un, proposals_un, im_hw=im_hw, ground_truth=False, com=True,alpha=float(beta),un=True)
            else:
                heatmaps_results_combined, combined_losses, heatmaps_combined = self.heatmaps(features, proposals_all, im_hw=im_hw, ground_truth=False, com=True,alpha=float(alpha),un=False)
      
        
        
        mse = 0
        
        if self.training:
            losses = {}
            if loss_type != "heatmap":
                #print("loss is not heatmap")
                losses.update(detector_losses)
                losses.update(proposal_losses)
            if loss_type == "combined" or loss_type == "combined_og":
                #print("Combo loss: ", combined_losses)
                #combined_losses['combined_loss'] = combined_losses['combined_loss'] * float(alpha)
                #print("in combo!")
                losses.update(combined_losses)
                if has_unlabeled == True:
                    losses.update(unlabeled_losses)
                   # print("UN loss:", unlabeled_losses)
               # print("DP loss: ", proposal_losses)
                #print("HM loss: ", heatmaps_losses)
                #print("Combo loss: ", combined_losses)
                #print("Un loss: ", unlabeled_losses)
                #pdb.set_trace()
                  
            losses.update(heatmaps_losses)
            #pdb.set_trace()
            return losses, mse
       
        c_loss_c = 0
        return result, heatmaps_results_all, stride, heatmaps_all, targets, c_loss_c
