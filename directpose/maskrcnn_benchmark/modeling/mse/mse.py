import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.backbone.deeplab import DLv3Plus
from maskrcnn_benchmark.layers import SigmoidFocalLoss
import math
import pdb
import numpy as np
from itertools import chain
import json
from maskrcnn_benchmark.structures.image_list import to_image_list


class Mse():
    def __init__(self):
        self.thresh = .2
        #self.cfg = cfg

    
    
def build_mse():
    return Mse()


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

def get_centroid(coord):
    return [(coord[0]+coord[2])/2, (coord[1]+coord[3])/2]

def get_data():
    #base = "../tools/datasets/bee/train/"
    base_val = "/home/bsb2144/directpose/tools/datasets/bee/annotations/train_bee_annotations2018_nondup.json"
    with open(base_val) as f:
        data_an = json.load(f)
    return data_an
        
def get_tps(predictions):
    ct = .2
    scores = predictions.get_field("scores")
    print("SCORES: ", scores)
    keep = torch.nonzero(scores > ct).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def get_dist(p1, p2):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return dist
        
#print(data_an["annotations"][0])

def get_mse(target, tps, file_id):
    mse_per_bee = []
    kps = tps.get_field("keypoints").keypoints
    bboxes = tps.bbox
    #print("TARGET bbox", bboxes)
    data_an = get_data()#file_id)
    #pdb.set_trace()
    #for p in data_an['images'][:]:
     #   if (str(p['file_name']) == test_files[idx] ):
      #      im_id = p['id']  
    im_id = file_id
    #print(im_id)
    for p in data_an['annotations'][:]:
        if ((p['image_id']) == im_id ):
            #print(p['id'])
            bb = p["segmentation"]#p['bbox']
            #bb = p['bbox']
            bb = bb[0] #comment line for test
            bb_f = [bb[:2], bb[2:4], bb[4:6], bb[6:]]
            bbox = bounding_box(bb_f)
            cent_val = get_centroid(bbox)
            dists = []
            #print(cent_val)
            for index, p_bbox in enumerate(bboxes):
                cp = get_centroid(p_bbox)
                #print("pbox: ", p_bbox)
                dist = get_dist(cent_val, cp)
                dists.append(dist)
                #print(dist)
            min_idx = dists.index(min(dists))
            #print("min id", min_idx)
            kps_valog = p["keypoints"]
           # print('kps pre del', kps_valog)
            #comment this line for test set
            kps_val = [x for i, x in enumerate(kps_valog) if (i+1)%3 !=0]
            #kps_val = kps_val[!(2::3)] 
            kps_pred = kps[min_idx].tolist()
            kps_pred_list = [kps_pred[0][:2],kps_pred[1][:2],kps_pred[2][:2],kps_pred[3][:2],kps_pred[4][:2]]
            kps_pl = list(chain.from_iterable(kps_pred_list))
            #print("kps val", kps_val)
            #print("kp pred", kps_pl)
            diff_vec = np.array(kps_val) - np.array(kps_pl)
            #print(diff_vec)
            #print(diff_vec*diff_vec)
            diff_vec_sum_sq = np.sum(diff_vec*diff_vec)
            #print(diff_vec_sum_sq)
            mse_per_bee.append(diff_vec_sum_sq)

    sum_mse_per_image = sum(mse_per_bee)
    mse_final = sum_mse_per_image/len(mse_per_bee)
    return mse_final
    
    
def get_mse_per_batch(targets, result, file_ids):
    mses_for_batch = []
    for (target, pred, file_id) in zip(targets, result, file_ids):
        top_predictions = get_tps(pred)
        if len(top_predictions.bbox.detach().cpu().numpy()) == 0:
            mses_for_batch.append(999999999)
            continue
            #FOR TEST USE ALL PREDS CHANGE BACK AFTER
        mse = get_mse(target, top_predictions, file_id)
        #mse = get_mse(target, pred, file_id)
        print("MSEEEE", mse)
        #print("TPS: ", top_predictions, len(top_predictions))
        mses_for_batch.append(mse)
    print("mse array: ", mses_for_batch)
    if len(mses_for_batch) != 0:
        return sum(mses_for_batch)/len(mses_for_batch)
    return 999999;
    
    