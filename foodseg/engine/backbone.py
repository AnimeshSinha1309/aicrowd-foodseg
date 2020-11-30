"""
Implements functions to conveniently run the Detectron2 models.
Sets up the config, and runs the training and validation loops.
"""

import os
import torch
import pandas as pd
import numpy as np
import cv2 as cv
import tqdm
import wandb

assert torch.cuda.is_available(), 'Switch on GPU Runtime'


class Backbone(torch.nn.Module):

    class ClassificationDataset(torch.utils.data.Dataset):
        """
        The Data Loader for the Pre-trainer engine
        """

        def __init__(self, working_dir, subset_dir):
            from sklearn.preprocessing import LabelEncoder
            self.dir = os.path.join(working_dir, 'classify', subset_dir)
            self.df = pd.read_csv(os.path.join(self.dir, 'data.csv'))
            self.encoder = LabelEncoder()
            self.df['label'] = self.encoder.fit_transform(self.df['class'])        
            self.images = self.df['image'].values
            self.labels = self.df['label'].values

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, key):
            image = cv.imread(os.path.join(self.dir, self.images[key]))
            image = cv.resize(image, (224, 224))
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image).float().cuda()
            label = torch.tensor(self.labels[key]).cuda()
            return image, label


    def __init__(self, trainer):
        """
        Extracts the backbone and plugs a dense head in front of it.

        :param trainer: The Detectron2 object
        """
        super(self, self).__init__()
        self.resnet = trainer.model.self.bottom_up
        self.vertical = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(3, 3))
        self.dense_1 = torch.nn.Linear(6400, 2048)
        self.dense_2 = torch.nn.Linear(2048, 1024)
        self.dense_3 = torch.nn.Linear(1024, 273)

    def forward(self, image):
        """
        Forward methods for the neural network

        :param image: input of shape (BATCH_SIZE, 224, 224, 3)
        :returns: the output of the classifier (BATCH_SIZE, 273)
        """
        net = self.resnet(image)['res5']
        net = self.vertical(net)
        net = net.view(net.shape[0], -1)
        net = self.dense_1(net)
        net = self.dense_2(net)
        net = self.dense_3(net)
        return net

    def fit(self, epochs, working_dir):
        """
        Trains the model

        :param epochs: int, number of steps to train for
        :param working_dir: str, the current working directory
        """
        train_dataset = ClassificationDataset(working_dir, 'train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = ClassificationDataset(working_dir, 'val')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        batch_losses, batch_accuracy = [], []
        val_batch_losses, val_batch_accuracy = [], []

        for epoch in range(epochs):
            self.train()
            total_loss, total_samples, total_correct = 0, 0, 0
            train_iterator = tqdm.tqdm(train_dataloader)
            for X, y in train_iterator:
                optimizer.zero_grad()
                out = self(X)
                loss = criterion(out, y)
                total_loss += loss.item()
                total_samples += y.shape[0]
                total_correct += (torch.max(out, 1)[1] == y).float().sum().item()
                loss.backward()
                optimizer.step()
                train_iterator.set_postfix(
                    loss=total_loss/total_samples, 
                    accuracy=total_correct/total_samples)
            batch_losses.append(total_loss/total_samples)
            batch_accuracy.append(total_correct/total_samples)
            wandb.log({'Backbone Training Accuracy': batch_accuracy[-1], 
                       'Backbone Training Loss': batch_losses[-1]})
            
            total_loss, total_samples, total_correct = 0, 0, 0
            with torch.no_grad():
                self.eval()
                val_iterator = tqdm.tqdm(val_dataloader)
                for X, y in val_iterator:
                    out = self(X)
                    loss = criterion(out, y)
                    total_loss += loss.item()
                    total_samples += y.shape[0]
                    total_correct += (torch.max(out, 1)[1] == y).float().sum().item()
                    val_iterator.set_postfix(
                        loss=total_loss/total_samples, 
                        accuracy=total_correct/total_samples)
                val_batch_losses.append(total_loss/total_samples)
                val_batch_accuracy.append(total_correct/total_samples)    
            wandb.log({'Backbone Validation Accuracy': val_batch_accuracy[-1], 
                       'Backbone Validation Loss': val_batch_losses[-1]})

