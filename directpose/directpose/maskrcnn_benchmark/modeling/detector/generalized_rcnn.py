# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch, gc
#import torch.cuda.memory_summary
from torch import nn
import pdb
import logging

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
            
        if self.training and has_unlabeled and loss_type!="combined":
            raise ValueError("Using unlabeled frames without using combined loss")
            
        images = to_image_list(images)
  
        im_hw = images.tensors.size(2), images.tensors.size(3)
        
        if self.training:
            if has_unlabeled == False:
                body_features = self.backbone.body(images.tensors)
                features = self.backbone.fpn(body_features)
                heatmaps_results_all, heatmaps_losses, heatmaps_all = self.heatmaps(features, 
                                                                        targets, im_hw=im_hw, 
                                                                        ground_truth=True,
                                                                          labeled = True)
                stride = im_hw[0] // heatmaps_all.size(2)
            else:
                body_features_la = self.backbone.body(images_la.tensors)
                features_la = self.backbone.fpn(body_features_la)
                body_features_un = self.backbone.body(images_un.tensors)
                features_un = self.backbone.fpn(body_features_un)
                
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
        #print('BODY FEATURES: ', body_features)  
        torch.cuda.empty_cache()
        #START TEST
        if self.training and loss_type == "combined":
            self.eval()
            print('START EVAL')
            height, width = images.tensors[0].shape[1], images.tensors[0].shape[1]
            if has_unlabeled == False:
                #get all new features
                body_features_new = self.backbone.body(images.tensors)
                features_new = self.backbone.fpn(body_features_new)
                
                #get all new heatmaps
                heatmaps_results_new, _ ,heatmaps_new, _ = self.heatmaps(features_new, None, im_hw=im_hw, 
                                                                                ground_truth=True,
                                                                                  labeled = True)
                #get all new proposals
                proposals_new = self.rpn(images, features_new, None,heatmaps_results_new, heatmaps_new,
                                                          stride,labeled=False)
                
                proposals_new = proposals_new[0]
                print("PN", proposals_new )
                proposals_resized = []
                for prop in proposals_new:
                    prop = prop.resize((width, height))
                    proposals_resized.append(prop)
                proposals_new = proposals_resized 
                
                
                to_delete = [body_features_new, features_new, heatmaps_results_new, _ ,heatmaps_new]
                del body_features_new
                del features_new
                del heatmaps_results_new
                del _
                del heatmaps_new
                del to_delete
                #print("BFN", body_features_new)
                torch.cuda.empty_cache()
            else:
                body_features_la_new = self.backbone.body(images_la.tensors)
                features_la_new = self.backbone.fpn(body_features_la_new)

                body_features_un_new = self.backbone.body(images_un.tensors)
                features_un_new = self.backbone.fpn(body_features_un_new)

                heatmaps_results_la_new, _ ,heatmaps_la_new, _ = self.heatmaps(features_la_new, None, im_hw=im_hw, 
                                                                                    ground_truth=True,labeled = True)

                heatmaps_results_un_new, _ ,heatmaps_un_new, _ = self.heatmaps(features_un_new, None, im_hw=im_hw, 
                                                                                    ground_truth=True,labeled = True)

                proposals_la_new = self.rpn(images_la, features_la_new, None,
                                                              heatmaps_results_la_new, heatmaps_la_new, stride,labeled=False)

                proposals_un_new = self.rpn(images_un, features_un_new, None, heatmaps_results_un_new, heatmaps_un_new,
                                                           stride,labeled=False)
                #print("prop un new", proposals_un_new)
                proposals_la_new = proposals_la_new[0]
                
                proposals_un_new = proposals_un_new[0]
                #print("prop un new 2", proposals_un_new)
                
                proposals_la_resized = []
                proposals_un_resized = []
                
                for prop_la in proposals_la_new:
                    prop_la = prop_la.resize((width, height))
                    proposals_la_resized.append(prop_la)
                    
                for prop_un in proposals_un_new:
                    prop_un = prop_un.resize((width, height))
                    proposals_un_resized.append(prop_un)
                    
                proposals_la_new = proposals_la_resized 
                proposals_un_new = proposals_un_resized 
                #print("prop un new 3", proposals_un_new)
               
                to_delete = ['body_features_la_new', 'body_features_un_new', 'features_la_new', 'features_un_new', 'heatmaps_results_la_new', 'heatmaps_results_un_new', '_' ,'heatmaps_la_new', 'heatmaps_un_new']
                for _var in to_delete:
                    globals().pop(_var, None)
                del to_delete
                torch.cuda.empty_cache()
         
            self.train()
            print('END EVAL')
        #END TEST
        torch.cuda.empty_cache()
        if self.training:
            if has_unlabeled == True:
                proposals_la, proposal_losses = self.rpn(images_la, 
                                                         features_la, 
                                                         targets_la,
                                                         heatmaps_results_la, 
                                                         heatmaps_la, stride,
                                                         labeled=True)
            else:  
                proposals_all, proposal_losses = self.rpn(images, features, targets, heatmaps_results_all, heatmaps_all, stride,labeled=True)

        if not self.training:
            proposals_all = self.rpn(images, features, targets, heatmaps_results_all, heatmaps_all, stride,labeled=False)
        
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            if self.training:
                # RPN-only models don't have roi_heads
                result = {}
                if loss_type == "combined":
                    if has_unlabeled == True:
                        result = proposals_la_new
                    else:
                        result = proposals_new
                    
                detector_losses = {}
                
        if not self.training: 
            result = proposals_all 
        
        # need to resize/edit proposals
        
        # get combined loss
        if loss_type == "combined":
            if has_unlabeled == True:
                #get combined loss for labeled images
                _, combined_losses, _ = self.heatmaps(features_la, proposals_la_new, im_hw=im_hw, ground_truth=False,
                                                      com=True,alpha=float(alpha),un=False)
                
                #get combined loss for unlabeled images
                _, unlabeled_losses, _ = self.heatmaps(features_un,proposals_un_new, im_hw=im_hw, 
                                                       ground_truth=False, com=True,alpha=float(beta), un=True)
            else:
                _, combined_losses, _ = self.heatmaps(features, proposals_new, im_hw=im_hw, ground_truth=False,
                                                     com=True,alpha=float(alpha), un=False)

        def select_top_predictions(predictions):
            """
            Select only predictions which have a `score` > self.confidence_threshold,
            and returns the predictions in descending order of score

            Arguments:
                predictions (BoxList): the result of the computation by the model.
                    It should contain the field `scores`.

            Returns:
                prediction (BoxList): the detected objects. Additional information
                    of the detection properties can be found in the fields of
                    the BoxList via `prediction.fields()`
            """
            scores = predictions.get_field("scores")
            keep = torch.nonzero(scores >= .2).squeeze(1)
            predictions = predictions[keep]
            scores = predictions.get_field("scores")
            _, idx = scores.sort(0, descending=True)
            return predictions[idx]

      
        logger = logging.getLogger("maskrcnn_benchmark.trainer")
        
        
        diffs = {'diffs':[], 'preds': [], 'imgs': []}
        if loss_type == "combined":
            if has_unlabeled:
                targets_iter =  targets_la
                proposals_iter = proposals_la_new
            else:
                targets_iter =  targets
                proposals_iter = proposals_new

            for target, prop, im in zip(targets_iter, proposals_iter, images_la.tensors):
                top_props = select_top_predictions(prop)
                #resize_top_props = []
                #prop = prop.resize((608, 608))
                resize_top_props = top_props.resize((images.tensors[0].shape[1], images.tensors[0].shape[1]))
                
                diff = len(target.get_field('keypoints').keypoints) - len(top_props.get_field('scores'))
#                 logger.info(
#                     ("""Num Targets: {}\n
#                       Difference labeled: {}"""
#                     ).format(
#                         len(target.get_field('keypoints').keypoints),
#                         diff))
#                 diffs['diffs'].append(diff)
#                 diffs['preds'].append(resize_top_props)
#                 diffs['imgs'].append(im)

        if self.training:
            losses = {}
            if loss_type != "heatmap":
                losses.update(detector_losses)
                losses.update(proposal_losses)
            if loss_type == "combined":
                losses.update(combined_losses)
                if has_unlabeled == True:
                    losses.update(unlabeled_losses)   
            losses.update(heatmaps_losses)
            return losses, diffs
       
        c_loss_c = 0
        return result, heatmaps_results_all, stride, heatmaps_all, targets, c_loss_c
