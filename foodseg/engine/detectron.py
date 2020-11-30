"""
Implements functions to conviniently run the Detectron2 models.
Sets up the config, and runs the training and validation loops.
"""

import os
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

setup_logger()
assert torch.cuda.is_available(), 'Switch on GPU Runtime'


class DetectronEngine:
    """
    Class to handle Config, Training, Evaluation, etc.
    """

    def __init__(self, working_dir, iterations = 5000, batch_size=16, num_classes=300):
        """
        Sets up the configuration for the Detectron model

        :param working_dir: str, directory to save outputs in (stored in working_dir/output)
        :param batch_size: int, batch size for backbone
        :param num_classes: upper bound on number of classes in dataset (number of dense heads)
        """
        self.train_annotations_path = os.path.join(
            working_dir, "data", "train", "processed_annotations.json")
        self.train_images_path = os.path.join(
            working_dir, "data", "train", "images/")
        self.val_annotations_path = os.path.join(
            working_dir, "data", "val", "processed_annotations.json")
        self.val_images_path = os.path.join(
            working_dir, "data", "val", "images/")

        register_coco_instances(
            "my_dataset_train", {}, self.train_annotations_path, self.train_images_path)
        register_coco_instances(
            "my_dataset_val", {}, self.val_annotations_path, self.val_images_path)

        self.cfg = get_cfg()
        # Check the model zoo and use any of the models ( from detectron2 github repo)
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ("my_dataset_val", )
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        self.cfg.OUTPUT_DIR = os.path.join(working_dir, 'output')
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.cfg.SOLVER.MAX_ITER = iterations
        self.trainer = DefaultTrainer(self.cfg)

    def train(self, resume):
        """
        Trains the model

        :param iterations: int, number of steps to train for
        :param resume: bool, if True, resume training an existing
                             model in the given output directory
        """
        self.trainer.resume_or_load(resume=resume)
        self.trainer.train()

    def evaluate(self):
        """
        Runs the evaluations for the model
        Returns the accuracy statistics
        """
        evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "my_dataset_val")
        val_results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
        return val_results
