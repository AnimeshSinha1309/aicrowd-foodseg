"""
Implements functions to conviniently run the Detectron2 models.
Sets up the config, and runs the training and validation loops.
"""

import torch, torchvision
assert torch.cuda.is_available(), 'Switch on GPU Runtime'

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from pycocotools.coco import COCO
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import pycocotools.mask as mask_util


def setup_config(iterations, working_dir, batch_size=16, num_classes=300):
    # Change the paths of the below mentioned directoryectories and files if you have made any changes.
    train_annotations_path = os.path.join(working_dir, "train/processed_annotations.json")
    train_images_path = os.path.join(working_dir, "train/images/")
    val_annotations_path = os.path.join(working_dir, "val/processed_annotations.json")
    val_images_path = os.path.join(working_dir, "val/images/")

    register_coco_instances("my_dataset_train", {}, train_annotations_path, train_images_path)
    register_coco_instances("my_dataset_val", {}, val_annotations_path, val_images_path)

    cfg = get_cfg()
    # Check the model zoo and use any of the models ( from detectron2 github repo)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.OUTPUT_directory = os.path.join(working_dir, 'outputs')
    os.makedirectorys(cfg.OUTPUT_directory, exist_ok=True)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_directory, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("my_dataset_val", )


def train(resume):
    cfg = get_cfg()
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=resume)
    # Training happens here
    trainer.train()

def evaluate():
    cfg = get_cfg()
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_directory=cfg.OUTPUT_directory)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    val_results = inference_on_dataset(trainer.model, val_loader, evaluator)

def predict():
    cfg = get_cfg()
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_directory=cfg.OUTPUT_directory)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    val_results = inference_on_dataset(trainer.model, val_loader, evaluator)
