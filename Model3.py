import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io

from detectron2 import model_zoo
from detectron2.config import get_cfg 
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

coco_dir='COCO_DIR'
img_dir = "img/"
train_dir = "train/"
val_dir = "val/"
test_dir = "test/"
if __name__ == '__main__':
    register_coco_instances("car_part_train", {}, os.path.join(coco_dir,train_dir,"COCO_mul_train_annos.json"), os.path.join(coco_dir,train_dir))
    register_coco_instances("car_part_val", {}, os.path.join(coco_dir,val_dir,"COCO_mul_val_annos.json"), os.path.join(coco_dir,val_dir))

    car_part_dataset_dicts = DatasetCatalog.get("car_part_train")
    car_part_metadata_dicts = MetadataCatalog.get("car_part_train")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file('configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.DATASETS.TRAIN = ("car_part_train",)
    cfg.DATASETS.TEST = ("car_part_val",)
    cfg.INPUT.CROP.ENABLED = True
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.WARMUP_ITERS = int(800)
    cfg.SOLVER.MAX_ITER = int(2500) #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (int(600), int(2000))
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(256)   # faster, and good enough for this  dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (damage)
    cfg.MODEL.RETINANET.NUM_CLASSES = 5 # only has one class (damage)
    cfg.TEST.EVAL_PERIOD = int(400)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(400)



    # Clear any logs from previous runs
    #TODO add timestamp to logs
    # !rm -rf cfg.OUTPUT_DIR
    # import shutil
    # if os.path.exists(cfg.OUTPUT_DIR):
    #     shutil.rmtree(cfg.OUTPUT_DIR)


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()