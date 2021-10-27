# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints, BeeKeypoints, FlyKeypoints, PupKeypoints, MonkeyKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, kfun="PersonKeypoints"
    ):
        print("name?")
        print(ann_file)
        print(root)
        print(kfun)
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        #print("IDSSS: ", self.ids)
        self.ids = sorted(self.ids)
        print("len ids pre ", len(self.ids))
        self.ann_file = ann_file
        if kfun=="PersonKeypoints":
            self.kfun = PersonKeypoints
        elif kfun=="BeeKeypoints":
            self.kfun = BeeKeypoints
        elif kfun=="FlyKeypoints":
            self.kfun = FlyKeypoints
        elif kfun=="PupKeypoints":
            self.kfun = PupKeypoints
        elif kfun=="MonkeyKeypoints":
            self.kfun = MonkeyKeypoints

        # filter images without detection annotations
        if False: #remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
            #print("ids: :", ids)
        #print("lid: ", len(self.ids))
            
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        #print("map: ", self.id_to_img_map)

    def compute_bboxes_for_keypoints(self, keypoints):
        INF = 100000000
        vis = keypoints[:, :, 2]
        _keypoints = keypoints[:, :, :2].clone()

        _keypoints[vis == 0] = INF
        bboxes_lt = (_keypoints.min(dim=1)[0] - 1).clamp(min=0.0)
        _keypoints[vis == 0] = -INF
        bboxes_rb = _keypoints.max(dim=1)[0] + 1

        return torch.cat([bboxes_lt, bboxes_rb], dim=1)

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        #masks = [obj["segmentation"] for obj in anno]
        #masks = SegmentationMask(masks, img.size, mode='poly')
        #target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            
            keypoints = self.kfun(keypoints, img.size)
            target.add_field("keypoints", keypoints)

            bboxes = self.compute_bboxes_for_keypoints(keypoints.keypoints)
            new_target = BoxList(bboxes, target.size, target.mode)
            for field in target.fields():
                new_target.add_field(field, target.get_field(field))
            target = new_target

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    def get_map(self):
        return self.id_to_img_map
    
