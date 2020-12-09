from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import random
import cv2
import numpy as np
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from os import listdir
from os.path import isfile, join
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

register_coco_instances("tiny_voc_test", {}, "./data/test.json ", "./data/test_images/test_images")

parser = argparse.ArgumentParser()
parser.add_argument("--test_model", default="model_final.pkl")
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)
cfg.DATASETS.TEST = ("tiny_voc_test",)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
cfg.MODEL.WEIGHTS = os.path.join('./output', args.test_model)
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("tiny_voc_test")

cocoGt = COCO("./data/test.json")
submit = []
for imgid in cocoGt.imgs:
    im = cv2.imread("./data/test_images/test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])
    outputs = predictor(im)
    score = outputs['instances'].scores
    score = (score.cpu().numpy()).tolist()
    label = outputs['instances'].pred_classes
    label = (label.cpu().numpy()).tolist()
    mask = outputs['instances'].pred_masks
    mask = (mask.cpu().numpy())
    n_instances = len(score)
    if len(label) > 0:
        for i in range(n_instances):
            pred = {}
            pred['image_id'] = imgid # this imgid must be same as the key of test.json
            pred['category_id'] = (int(label[i]) + 1)
            pred['segmentation'] = binary_mask_to_rle(mask[i]) # save binary mask to RLE, e.g. 512x512 -> rle
            pred['score'] = float(score[i])
            submit.append(pred)

with open("0756545.json", "w") as f:
    json.dump(submit, f)