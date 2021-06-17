import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.backbone.deeplab import DLv3Plus
from maskrcnn_benchmark.layers import SigmoidFocalLoss
import math


class Heatmaps(nn.Module):
    def __init__(self, cfg, in_channels_list):
        super(Heatmaps, self).__init__()
        if cfg.DATATYPE=='person':
            self.num_kps = 17
        elif cfg.DATATYPE=='bee':
            self.num_kps = 5
        elif cfg.DATATYPE=='fly':
            self.num_kps = 13
            
        self.deeplab = DLv3Plus(cfg, self.num_kps, in_channels_list)
        self.heatmaps_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.heatmaps_loss_weight = cfg.MODEL.HEATMAPS_LOSS_WEIGHT

        for modules in [self.deeplab]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.deeplab.heatmaps.bias, bias_value)

    def get_local_max_coords(self, heatmaps, kernel_size=3):
        assert kernel_size % 2 == 1
        max_heatmaps = F.max_pool2d(
            heatmaps,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        coords = torch.nonzero(
            (max_heatmaps == heatmaps) & (torch.sigmoid(heatmaps) > 0.05)
        )
        return coords

    def forward(self, features, targets=None, im_hw=None):
        heatmaps = self.deeplab(features)
        stride = im_hw[0] // heatmaps.size(2)
        if self.training:
            heatmaps_loss = self._forward_train(heatmaps, targets, stride)
            return None, {"heatmaps_loss": heatmaps_loss * self.heatmaps_loss_weight}, heatmaps
        else:
            coords = self.get_local_max_coords(heatmaps)

            results = []
            for im_i in range(features[0].size(0)):
                coords_per_im = coords[coords[:, 0] == im_i]
                results_per_type = []
                for type_i in range(self.num_kps):
                    coords_per_type = coords_per_im[coords_per_im[:, 1] == type_i][:, -2:].float() * stride
                    results_per_type.append(coords_per_type[:, [1, 0]])
                results.append(results_per_type)

            # import cv2
            # heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min())
            # # for i in range(heatmaps[0].size(0)):
            # heatmaps_im = heatmaps[0].max(dim=0)[0].cpu().numpy()
            # heatmaps_im = cv2.resize(heatmaps_im, (0, 0), fx=1, fy=1)
            # cv2.imshow("heatmaps_{}".format(0), heatmaps_im)
            # cv2.waitKey()

            return results, stride, heatmaps

    def _forward_train(self, heatmaps, targets, stride):
        heatmap_hw = heatmaps.size(2), heatmaps.size(3)
        gt_kps_heatmaps = self.prepare_kps_targets(
            targets,
            stride,
            heatmap_hw
        )

        # import cv2
        # gt_heatmaps_im = gt_kps_heatmaps[0].max(dim=0)[0].cpu().numpy()
        # gt_heatmaps_im = cv2.resize(gt_heatmaps_im, (0, 0), fx=4, fy=4)
        # cv2.imshow("gt_kps_heatmaps", gt_heatmaps_im)
        # cv2.waitKey()

        heatmaps = heatmaps.permute(0, 2, 3, 1).reshape(-1, self.num_kps)
        gt_kps_heatmaps = gt_kps_heatmaps.reshape(-1)
        num_pos = (gt_kps_heatmaps > 0).sum().item()

        heatmaps_loss = self.heatmaps_loss_func(
            heatmaps,
            gt_kps_heatmaps.int()
        ) / max(float(num_pos), 1.0)

        return heatmaps_loss

    def generate_kps_heatmaps_target(
            self, kps, kps_vis,
            num_kps_per_instance,
            stride, heatmap_hw
    ):
        '''
        :param joints:  [num_joints, 17, 2]
        :param joints_vis: [num_joints, 17]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        kps = kps.reshape(-1, 2)
        kps_vis = kps_vis.reshape(-1).clone()

        heatmap_h, heatmap_w = heatmap_hw
        target = kps.new_zeros((num_kps_per_instance, heatmap_h, heatmap_w))

        sigma = 2
        tmp_size = sigma * 3

        for kp_id in range(len(kps)):
            kp_type = kp_id % num_kps_per_instance

            mu_x = int(kps[kp_id][0] / stride + 0.5)
            mu_y = int(kps[kp_id][1] / stride + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = (int(mu_x - tmp_size), int(mu_y - tmp_size))
            br = (int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1))
            if ul[0] >= heatmap_w or ul[1] >= heatmap_h \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                kps_vis[kp_id] = 0
                continue

            if kps_vis[kp_id] == 0:
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = torch.arange(size, dtype=torch.float32, device=kps.device)
            y = x[:, None]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_h) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_w)
            img_y = max(0, ul[1]), min(br[1], heatmap_h)

            local_target = torch.max(
                target[kp_type][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )
            target[kp_type][img_y[0]:img_y[1], img_x[0]:img_x[1]] = local_target

        # show_target(target.cpu().numpy())

        return target

    def generate_kps_label_maps_target(
            self, kps, kps_vis,
            stride, heatmap_hw
    ):
        '''
        :param joints:  [num_joints, 17, 2]
        :param joints_vis: [num_joints, 17]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_kps_per_instance = kps.size(1)
        kps = kps.reshape(-1, 2)
        kps_types = torch.arange(kps.size(0), dtype=kps.dtype, device=kps.device) % num_kps_per_instance + 1
        kps_vis = kps_vis.reshape(-1).clone()
        heatmap_h, heatmap_w = heatmap_hw
        target = kps.new_zeros(heatmap_h * heatmap_w, dtype=torch.float32)
        RANGE = 1 * stride

        visible_kps = kps[kps_vis > 0]
        visible_kps_types = kps_types[kps_vis > 0]
        if len(visible_kps) == 0:
            return target.reshape(heatmap_h, heatmap_w)

        locations = self.compute_locations_per_level(heatmap_h, heatmap_w, stride, visible_kps.device)
        locations_x = locations[:, 0][:, None]
        locations_y = locations[:, 1][:, None]
        distances_x = torch.abs(locations_x - visible_kps[:, 0][None])
        distances_y = torch.abs(locations_y - visible_kps[:, 1][None])
        distances = torch.max(distances_x, distances_y)

        min_distances, min_inds = distances.min(dim=0)
        target[min_inds] = visible_kps_types

        target = target.reshape(heatmap_h, heatmap_w)

        # import cv2
        # target_im = (target - target.min()) / (target.max() - target.min())
        # cv2.imshow("target", target_im.cpu().numpy())
        # cv2.waitKey()

        return target

    def prepare_kps_targets(self, targets, stride, heatmap_hw):
        kps_heatmaps = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            keypoints = targets_per_im.get_field("keypoints").keypoints
            kps_heatmaps_per_im = self.generate_kps_label_maps_target(
                keypoints[:, :, :2],
                keypoints[:, :, 2],
                stride,
                heatmap_hw
            )
            kps_heatmaps.append(kps_heatmaps_per_im)
        return torch.stack(kps_heatmaps, dim=0)

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


def build_heatmaps(cfg, in_channels_list):
    return Heatmaps(cfg, in_channels_list)
