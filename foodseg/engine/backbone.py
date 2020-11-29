"""
Implements functions to conveniently run the Detectron2 models.
Sets up the config, and runs the training and validation loops.
"""

import os
import torch

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

assert torch.cuda.is_available(), 'Switch on GPU Runtime'


class Backbone(torch.nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        trainer = DefaultTrainer(get_cfg())
        self.resnet = trainer.model.backbone.bottom_up
        self.vertical = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(3, 3))
        self.dense_1 = torch.nn.Linear(6400, 2048)
        self.dense_2 = torch.nn.Linear(2048, 1024)
        self.dense_3 = torch.nn.Linear(1024, 273)

    def forward(self, image):
        net = self.resnet(image)['res5']
        net = self.vertical(net)
        net = net.view(-1)
        net = self.dense_1(net)
        net = self.dense_2(net)
        net = self.dense_3(net)
        return net

    def fit(self, iterations, training_data, validation_data=None):
        """
        Trains the model

        :param iterations: int, number of steps to train for
        :param training_data: iterator, over all the training images
        :param validation_data: iterator, over all the validation images
        """
        raise NotImplementedError('Training is not implemented yet')

    def evaluate(self, validation_data):
        """
        Runs the evaluations for the model
        Returns the accuracy statistics
        :param validation_data: iterator, over all the validation images
        """
        raise NotImplementedError('Evaluation is not implemented yet')
