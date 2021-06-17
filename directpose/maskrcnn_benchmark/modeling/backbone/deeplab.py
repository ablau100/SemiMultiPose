import torch
from torch import nn
import torch.nn.functional as F

# from maskrcnn_benchmark.layers import SegLoss
from maskrcnn_benchmark.modeling import registry


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rates):
        super(ASPP_module, self).__init__()
        self.atrous_convs = nn.ModuleList()
        for i, rate in enumerate(rates):
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, inplanes, kernel_size=3,
                              stride=1, padding=rate, groups=inplanes,
                              dilation=rate, bias=False),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True)))

        self.__init_weight()

    def forward(self, x):
        outs = []
        for atrous_conv in self.atrous_convs:
            outs.append(atrous_conv(x))
        return torch.cat(outs, dim=1)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DLv3Plus(nn.Module):
    def __init__(self, cfg, num_classes, in_channels_list):
        super(DLv3Plus, self).__init__()
        self.do_aspp = True
        self.has_ins = True
        rates = (6, 12, 18)
        planes = 128
        inplanes, low_level_inplanes = in_channels_list[3], in_channels_list[0]

        # if self.do_aspp:
        #     self.aspp = ASPP_module(inplanes, planes, rates)
        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(inplanes, planes, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(planes),
        #     nn.ReLU(inplace=True))
        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(inplanes, planes,
        #                                                1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(planes),
        #                                      nn.ReLU())
        # if self.do_aspp:
        #     self.conv1 = nn.Sequential(
        #         nn.Conv2d((2 + len(rates)) * planes, planes,
        #                   kernel_size=1, bias=False),
        #         nn.BatchNorm2d(planes),
        #         nn.ReLU(inplace=True))
        # else:
        #     self.conv1 = nn.Sequential(
        #         nn.Conv2d(2 * planes, planes, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(planes),
        #         nn.ReLU(inplace=True))
        #
        # # adopt [1x1, 48] for l4
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.last_conv = nn.Sequential(nn.Conv2d(planes + 48, planes, kernel_size=3, stride=1, padding=1, bias=False),
        #                                nn.BatchNorm2d(planes),
        #                                nn.ReLU(),
        #                                nn.Conv2d(
        #                                    planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
        #                                nn.BatchNorm2d(planes),
        #                                nn.ReLU())

        self.seg_head = nn.Sequential(nn.Conv2d(low_level_inplanes, planes, kernel_size=3,
                                                stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(),
                                      nn.Conv2d(planes, planes, kernel_size=3,
                                                stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU())
        self.heatmaps = nn.Conv2d(planes, num_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # self.loss_on = cfg.MODEL.ALIGN.PANOPTIC_ON
        # if self.loss_on:
        #     thing_classes = cfg.MODEL.PANOPTIC.THING_CLASSES + 1
        #     stuff_classes = cfg.MODEL.PANOPTIC.STUFF_CLASSES
        #     self.lambda_s = cfg.MODEL.PANOPTIC.LAMBDA_S
        #     self.lambda_i = cfg.MODEL.PANOPTIC.LAMBDA_I
        #     self.loss = SegLoss(min_cls=stuff_classes - 1,
        #                         scale_factor=1 / 8)
        #     self.seg_head = nn.Sequential(nn.Conv2d(planes + 48, planes, kernel_size=3,
        #                                             stride=1, padding=1, bias=False),
        #                                   nn.BatchNorm2d(planes),
        #                                   nn.ReLU(),
        #                                   nn.Conv2d(planes, planes, kernel_size=3,
        #                                             stride=1, padding=1, bias=False),
        #                                   nn.BatchNorm2d(planes),
        #                                   nn.ReLU(),
        #                                   nn.Conv2d(planes, thing_classes, kernel_size=1,
        #                                             stride=1))
        #     self.ins_head = nn.Sequential(nn.Conv2d(planes + 48, planes, kernel_size=3,
        #                                             stride=1, padding=1, bias=False),
        #                                   nn.BatchNorm2d(planes),
        #                                   nn.ReLU(),
        #                                   nn.Conv2d(planes, planes, kernel_size=3,
        #                                             stride=1, padding=1, bias=False),
        #                                   nn.BatchNorm2d(planes),
        #                                   nn.ReLU(),
        #                                   nn.Conv2d(planes, 3, kernel_size=1,
        #                                             stride=1),
        #                                   nn.Tanh())

    def forward(self, feats, targets=None):
        l1, l2, l3, l4, l5 = feats
        return self.heatmaps(self.seg_head(l1))
        # l1, l2, l3, l4 = feats
        # low_level_backbone_features = l1
        # if self.do_aspp:
        #     aspp = self.aspp(l4)
        # x1 = self.conv1x1(l4)
        # x5 = self.global_avg_pool(l4)
        # x5 = F.interpolate(x5, size=l4.size()[
        #                             2:], mode='bilinear', align_corners=False)
        #
        # if self.do_aspp:
        #     x = torch.cat((x1, aspp, x5), dim=1)
        # else:
        #     x = torch.cat((x1, x5), dim=1)
        # x = self.conv1(x)
        # x = F.interpolate(
        #     x, size=low_level_backbone_features.size()[2:],
        #     mode='bilinear', align_corners=False
        # )
        #
        # low_level_features = self.conv2(low_level_backbone_features)
        #
        # x = torch.cat((x, low_level_features), dim=1)
        # base_out = self.heatmaps(self.last_conv(x))
        #
        # return base_out

        # losses = {}
        # if self.training and self.loss_on:
        #     seg_out = self.seg_head(x)
        #     ins_out = self.ins_head(x)
        #     loss_dict = self.loss(seg_out, targets, ins_out)
        #     losses['loss_segm'] = loss_dict['seg'] * self.lambda_s
        #     losses['loss_inst'] = loss_dict['ins'] * self.lambda_i
        #     losses['loss_bg'] = loss_dict['bg']
        #
        # return base_out, losses


# @registry.PANOPTIC_HEADS.register("dlv3plus")
# def build_deeplab(cfg):
#     return DLv3Plus(cfg)
#
#
# @registry.PANOPTIC_HEADS.register("UPS")
# def build_ups(cfg):
#     return EmptyModule()


class EmptyModule(nn.Module):
    def forward(self, *args):
        return ([None], [None]), {}
