import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.keypoint import Keypoints


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        centroid,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.centroid = centroid
        
    def forward_for_single_feature_map(
            self, locations, box_cls,
            centerness, kps_pred,
            image_sizes, centroid_layer, hm_sigmoid, stride
    ):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        Hs, Ws = hm_sigmoid.size(2), hm_sigmoid.size(3)
        N, C, H, W = box_cls.shape
        N_kps = kps_pred.size(1) // 2

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        # box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        # box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        kps_pred = kps_pred.view(N, -1, H, W).permute(0, 2, 3, 1)
        kps_pred = kps_pred.reshape(N, -1, N_kps, 2)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        #box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            # per_box_regression = box_regression[i]
            # per_box_regression = per_box_regression[per_box_loc]
            per_kps_pred = kps_pred[i]
            per_kps_pred = per_kps_pred[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                # per_box_regression = per_box_regression[top_k_indices]
                per_kps_pred = per_kps_pred[top_k_indices]
                per_locations = per_locations[top_k_indices]

            # detections = torch.stack([
            #     per_locations[:, 0] - per_box_regression[:, 0],
            #     per_locations[:, 1] - per_box_regression[:, 1],
            #     per_locations[:, 0] + per_box_regression[:, 2],
            #     per_locations[:, 1] + per_box_regression[:, 3],
            # ], dim=1)

            per_kps_pred = per_kps_pred.view(-1, N_kps, 2) + per_locations[:, None, :]
            
#             per_kps_pred_hm = (per_kps_pred/stride).to(dtype=torch.int64)
#             print('per_kps_pred_hm.shape',per_kps_pred_hm.shape)
#             print('per_kps_pred_hm',per_kps_pred_hm.max())
#             print('hm_sigmoid',hm_sigmoid.shape)
#             cl = 0
#             for j in range(N_kps):
#                 hm_i = hm_sigmoid[i,j,:,:]
#                 kpx = per_kps_pred_hm[:,j,0]
#                 kpy = per_kps_pred_hm[:,j,1]
#                 kpx = torch.where(kpx<Ws, kpx, torch.tensor(Ws-1,device=kpx.device))
#                 kpy = torch.where(kpy<Hs, kpy, torch.tensor(Hs-1,device=kpy.device))
#                 cl += hm_i[kpx,kpy]
#             cl = cl/N_kps
            
            per_kps_v = per_kps_pred.new_ones((per_kps_pred.size(0), per_kps_pred.size(1), 1))
            per_kps_pred = torch.cat([per_kps_pred, per_kps_v], dim=-1)

            if len(per_kps_pred) > 0:
                bbox_lt = per_kps_pred[:, :, :2].min(dim=1)[0]
                bbox_rb = per_kps_pred[:, :, :2].max(dim=1)[0]
                detections = torch.cat([bbox_lt, bbox_rb], dim=1)
            else:
                detections = per_kps_pred.new_empty((0, 4))
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("keypoints", per_kps_pred)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward_for_single_feature_map_centroids(
            self, locations, box_cls,
            centerness, kps_pred,
            image_sizes, centroid_layer, hm_sigmoid, stride
    ):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        Hs, Ws = hm_sigmoid.size(2), hm_sigmoid.size(3)
        N, C, H, W = box_cls.shape
        N_kps = kps_pred.size(1) // 2

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        # box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        # box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        kps_pred = kps_pred.view(N, -1, H, W).permute(0, 2, 3, 1)
        kps_pred = kps_pred.reshape(N, -1, N_kps, 2)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # box_cls = box_cls * centerness[:, :, None]

        top_centroid = centroid_layer.to(dtype=torch.int64)
        top_centroid = top_centroid[:,0]+top_centroid[:,1]*W
        print('top_centroid',top_centroid)

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_box_cls = per_box_cls[top_centroid][:,0]

            per_box_loc = top_centroid
            per_class = top_centroid*0+1

            per_kps_pred = kps_pred[i]
            per_kps_pred = per_kps_pred[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            per_kps_pred = per_kps_pred.view(-1, N_kps, 2) + per_locations[:, None, :]
            
#             per_kps_pred_hm = (per_kps_pred/stride).to(dtype=torch.int64)
#             cl = 0
#             for j in range(N_kps):
#                 hm_i = hm_sigmoid[i,j,:,:]
#                 kpx = per_kps_pred_hm[:,j,0]
#                 kpy = per_kps_pred_hm[:,j,1]
#                 kpx = torch.where(kpx<Ws, kpx, torch.tensor(Ws-1,device=kpx.device))
#                 kpy = torch.where(kpy<Hs, kpy, torch.tensor(Hs-1,device=kpy.device))
#                 cl += hm_i[kpx,kpy]
#             cl = cl/N_kps
            
            per_kps_v = per_kps_pred.new_ones((per_kps_pred.size(0), per_kps_pred.size(1), 1))
            per_kps_pred = torch.cat([per_kps_pred, per_kps_v], dim=-1)

            if len(per_kps_pred) > 0:
                bbox_lt = per_kps_pred[:, :, :2].min(dim=1)[0]
                bbox_rb = per_kps_pred[:, :, :2].max(dim=1)[0]
                detections = torch.cat([bbox_lt, bbox_rb], dim=1)
            else:
                detections = per_kps_pred.new_empty((0, 4))
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("keypoints", per_kps_pred)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(
            self, locations, box_cls,
            box_regression, centerness,
            kps_pred, heatmaps_coords, heatmaps, image_sizes, stride
    ):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        
        centroid = heatmaps_coords[0][1]     
        hm_sigmoid = heatmaps.sigmoid()
        sampled_boxes = []
        bundle = zip(locations, box_cls, centerness, kps_pred)
        layers = 0
        for layers, (l, o, c, k) in enumerate(bundle):
            centroid_layer = centroid/stride/2**layers
            if self.centroid:
                sampled_boxes.append(
                    self.forward_for_single_feature_map_centroids(
                        l, o, c, k, image_sizes, centroid_layer, hm_sigmoid, stride
                    )
                )
            else:
                sampled_boxes.append(
                    self.forward_for_single_feature_map(
                        l, o, c, k, image_sizes, centroid_layer, hm_sigmoid, stride
                    )
                )
        
        if self.centroid:
            boxlists = self.select_over_all_levels_centroids(sampled_boxes)
        else:
            boxlists = list(zip(*sampled_boxes))
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists#, sampled_boxes

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels_centroids(self, bboxes):
        boxlist1 = 0
        boxlist2 = 0
        scores1_max = -100
        scores2_max = -100
        for bb in bboxes:
            bbox = bb[0].bbox
            labels = bb[0].get_field('labels')
            scores = bb[0].get_field('scores')
            keypoints = bb[0].get_field('keypoints')

            bbox1 = bbox[0,None,:]
            bbox2 = bbox[1,None,:]
            labels1 = labels[0,None]
            labels2 = labels[1,None]
            keypoints1 = keypoints[0,None,:,:]
            keypoints2 = keypoints[1,None,:,:]
            scores1 = scores[0,None]
            scores2 = scores[1,None]

            if scores1>scores1_max:
                bboxlist1 = BoxList(bbox1, (int(bb[0].size[0]), int(bb[0].size[1])), mode="xyxy")
                bboxlist1.add_field('labels',labels1)
                bboxlist1.add_field('scores',scores1)
                bboxlist1.add_field('keypoints',keypoints1)
                scores1_max = scores1
            if scores2>scores2_max:
                bboxlist2 = BoxList(bbox2, (int(bb[0].size[0]), int(bb[0].size[1])), mode="xyxy")
                bboxlist2.add_field('labels',labels2)
                bboxlist2.add_field('scores',scores2)
                bboxlist2.add_field('keypoints',keypoints2)   
                scores2_max = scores2

        boxlist = [[bboxlist1]]+[[bboxlist2]]
        boxlist = list(zip(*boxlist))
        boxlist = [cat_boxlist(bb) for bb in boxlist]
        return boxlist

    
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            keypoints = boxlists[i].get_field("keypoints")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                keypoints_j = keypoints[inds]
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class.add_field(
                    "keypoints",
                    Keypoints(keypoints_j, boxlist.size)
                )
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        centroid=config.MODEL.CENTROID
    )

    return box_selector
