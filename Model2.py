import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.config import CfgNode as CN
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Set base params
plt.rcParams["figure.figsize"] = [16, 9]

import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image

coco_dir = 'COCO_DIR'
img_dir = "img/"
train_dir = "train/"
val_dir = "val/"
test_dir = "test/"

if __name__ == '__main__':
    register_coco_instances("damage_type_train", {}, os.path.join(coco_dir, train_dir, "COCO_damage_severity.json"), os.path.join(coco_dir, train_dir))
    register_coco_instances("damage_type_val", {}, os.path.join(coco_dir, val_dir, "COCO_damage_annos.json"), os.path.join(coco_dir, val_dir))

    damage_type_dataset_dicts = DatasetCatalog.get("damage_type_train")
    damage_type_metadata_dicts = MetadataCatalog.get("damage_type_train")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("damage_type_train",)
    cfg.DATASETS.TEST = ("damage_type_val",)
    cfg.INPUT.CROP.ENABLED = True
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.WARMUP_ITERS = int(800)
    cfg.SOLVER.MAX_ITER = int(3500)  # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (int(600), int(2000), int(3250))
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(256)  # faster, and good enough for this dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(3)  # only has one class (damage)
    cfg.MODEL.RETINANET.NUM_CLASSES = int(3)  # only has one class (damage)
    cfg.TEST.EVAL_PERIOD = int(400)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(400)

    # cfg.MODEL.DEVICE = "cpu"

    # Clear any logs from previous runs
    # TODO add timestamp to logs
    # import shutil
    # if os.path.exists(cfg.OUTPUT_DIR):
    #     shutil.rmtree(cfg.OUTPUT_DIR)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()