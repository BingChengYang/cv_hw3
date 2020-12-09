from detectron2.data.datasets import register_coco_instances
import random
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
import argparse
register_coco_instances("tiny_voc", {}, "./data/pascal_train.json ", "./data/train_images/train_images")

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.00015)
parser.add_argument("--mini_batch_size", default=32)
parser.add_argument("--epoches", default=5000)
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("tiny_voc",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = args.lr
cfg.SOLVER.MAX_ITER = (
    args.epoches
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    args.mini_batch_size
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()