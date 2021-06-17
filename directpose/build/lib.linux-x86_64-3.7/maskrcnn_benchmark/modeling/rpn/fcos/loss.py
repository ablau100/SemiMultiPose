"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import maskrcnn_benchmark.engine.trainer as trainer
from maskrcnn_benchmark.config import cfg


INF = 100000000


def bilog(x):
    return torch.where(x >= 0, torch.log(x + 1), -torch.log(1 - x))


class OKSLoss(nn.Module):
    def forward(self, pred, target, v_target, weight=None):
        '''
        :param pred: N x 34 x 2
        :param target: N x 17 x 2
        :param v_target: N x 17
        :param weight: N
        :return:
        '''
        N = pred.size(0)
        N_kps = target.size(1)
        pred = pred.reshape(-1, 2)
        target = target.reshape(-1, 2)
        v_target = v_target.reshape(-1)

        # bases = .reshape(-1, 2)

        # pred = bilog(pred)
        losses = torch.abs(pred - target)
        # dfs_losses1 = torch.abs(offsets1 + bases - target)
        # dfs_losses2 = torch.abs(offsets2 + offsets1 + bases - target)
        #
        # if trainer.global_step <= cfg.SOLVER.MAX_ITER / 3:
        #     losses = conv_losses
        # elif trainer.global_step <= (cfg.SOLVER.MAX_ITER * 2) / 3:
        #     losses = dfs_losses1
        # else:
        #     losses = dfs_losses2

        losses = losses.mean(dim=-1)

        losses = losses.reshape(N, N_kps)
        v_target = v_target.reshape(N, N_kps)
        v_target = (v_target > 0).float()
        losses = losses * v_target
        losses = losses.sum(dim=-1) / v_target.sum(dim=-1).clamp(min=1.0)
        v_target_instance = (v_target.sum(dim=-1) > 0).float()

        if weight is not None:
            loss = (losses * weight * v_target_instance).sum() / \
                   (weight * v_target_instance).sum().clamp(min=1.0)
        else:
            loss = losses.mean()
        return loss


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.kps_loss_func = OKSLoss()
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets, fpn_strides):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        expanded_fpn_strides = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
            fpn_strides_per_level = points_per_level.new_tensor([fpn_strides[l]])
            expanded_fpn_strides.append(
                fpn_strides_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        expanded_fpn_strides = torch.cat(expanded_fpn_strides, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, kps_targets = self.compute_targets_for_locations(
            points_all_level, targets,
            expanded_object_sizes_of_interest,
            expanded_fpn_strides
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            kps_targets[i] = torch.split(kps_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        kps_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )
            kps_targets_level_first.append(
                torch.cat([kps_targets_per_im[level] for kps_targets_per_im in kps_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first, kps_targets_level_first

    def compute_targets_for_locations(
            self, locations,
            targets, object_sizes_of_interest,
            expanded_fpn_strides):
        labels = []
        reg_targets = []
        kps_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            keypoints = targets_per_im.get_field("keypoints").keypoints
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = self.get_sample_region(
                bboxes,
                self.fpn_strides,
                self.num_points_per_level,
                xs,
                ys,
                radius=1.5
            )

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_aera == INF] = 0
            keypoints = keypoints[locations_to_gt_inds]

            kps_xy_targets_per_im = keypoints[:, :, :2] - locations.view(-1, 1, 2)
            # normalize by fpn_strides
            kps_xy_targets_per_im = kps_xy_targets_per_im / \
                                    expanded_fpn_strides.reshape(-1, 1, 1).float()
            kps_v_targets_per_im = keypoints[:, :, 2]
            kps_xy_targets_per_im[kps_v_targets_per_im == 0] = 0

            kps_targets_per_im = torch.cat([
                kps_xy_targets_per_im,
                kps_v_targets_per_im[:, :, None]
            ], dim=2)

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            kps_targets.append(kps_targets_per_im)

        return labels, reg_targets, kps_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, kps_pred, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        
        N = box_cls[0].size(0)
        N_kps = kps_pred[0].size(1) // 2
        
        num_classes = box_cls[0].size(1)
        labels, reg_targets, kps_targets = self.prepare_targets(locations, targets, self.fpn_strides)

        box_cls_flatten = []
        # box_regression_flatten = []
        centerness_flatten = []
        kps_pred_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        kps_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            # box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
            kps_pred_flatten.append(kps_pred[l].permute(0, 2, 3, 1).reshape(-1, N_kps, 2))

            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            kps_targets_flatten.append(kps_targets[l].reshape(-1, N_kps, 3))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        # box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        kps_pred_flatten = torch.cat(kps_pred_flatten, dim=0)

        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        kps_targets_flatten = torch.cat(kps_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        # box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        kps_pred_flatten = kps_pred_flatten[pos_inds]

        kps_targets_flatten = kps_targets_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            # reg_loss = self.box_reg_loss_func(
            #     box_regression_flatten,
            #     reg_targets_flatten,
            #     centerness_targets
            # )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
            kps_loss = self.kps_loss_func(
                kps_pred_flatten,
                kps_targets_flatten[:, :, :2],
                kps_targets_flatten[:, :, 2],
                centerness_targets
            )
        else:
            # reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            kps_loss = kps_pred_flatten.sum()

        return cls_loss, centerness_loss, kps_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
