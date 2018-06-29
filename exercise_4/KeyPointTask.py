import sys

from exercise_code.solver_keypoint import SolverKeyPoint

sys.path.append('../../')
import TaskPlan
import torch
from torchvision import transforms
import tensorflow as tf
from exercise_code.dataloader import FacialKeypointsDataset
import numpy as np
from exercise_code.classifiers.keypoint_nn import KeypointModel
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

class Normalize(object):
    """Normalizes keypoints.
    """

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################
        image = image.astype(np.float) / 255
        key_pts = (key_pts.astype(np.float) / 96) * 2 - 1
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}


class KeyPointTask(TaskPlan.Task):

    def __init__(self, preset, preset_pipe, logger, subtask):
        super().__init__(preset, preset_pipe, logger, subtask)

        # order matters! i.e. rescaling should come before a smaller crop
        data_transform = transforms.Compose([Normalize(), ToTensor()])

        # create the transformed dataset
        transformed_dataset = FacialKeypointsDataset(csv_file='datasets/training.csv', transform=data_transform)

        VAL_dataset = FacialKeypointsDataset(csv_file='datasets/val.csv', transform=data_transform)
        self.val_loader = DataLoader(VAL_dataset, batch_size=self.preset.get_int("batch_size"), shuffle=True, num_workers=4)

        self.model = KeypointModel()
        self.logger.log(str(self.model))
        self.solver = SolverKeyPoint(optim_args={'lr': self.preset.get_float("learning_rate"), 'weight_decay': self.preset.get_float("weight_decay")}, loss_func=nn.MSELoss())
        self.solver.set_model(self.model)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
        self.train_loader = DataLoader(transformed_dataset, batch_size=self.preset.get_int("batch_size"), shuffle=True, num_workers=4)
        self.train_iterator = iter(self.train_loader)

    def save(self, path):
        self.model.save(str(path / "keypoints_nn.model"))

    def step(self, tensorboard_writer, current_iteration):
        try:
            loss, metric = self.solver.step(self.model, self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            loss, metric = self.solver.step(self.model, self.train_iterator)

        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/training", simple_value=loss)]), current_iteration)
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="metric/training", simple_value=metric)]), current_iteration)

        if current_iteration % self.preset.get_int('val_interval') == 0:
            val_loss, val_metric = self.solver.validate(self.model, self.val_loader)
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/val", simple_value=val_loss)]), current_iteration)
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="metric/val", simple_value=val_metric)]), current_iteration)

    def load(self, path):
        self.model = torch.load(str(path / "keypoints_nn.model"))
        self.solver.set_model(self.model)
