import sys
sys.path.append('../../')
import TaskPlan
from exercise_code.classifiers.classification_cnn import ClassificationCNN
from exercise_code.solver import Solver
from exercise_code.data_utils import get_CIFAR10_datasets
import torch
from torch.autograd import Variable
from torchvision import transforms
import pickle
import tensorflow as tf

class Task(TaskPlan.Task):

    def __init__(self, preset, logger, subtask):
        super().__init__(preset, logger, subtask)

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(self.preset.get_float('rotate'), (self.preset.get_float('translate'), self.preset.get_float('translate')), (self.preset.get_float('scale_min'), self.preset.get_float('scale_max'))),
            transforms.ToTensor()
        ])

        train_data, val_data, test_data, mean_image = get_CIFAR10_datasets(transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.preset.get_int('batch_size'), shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.preset.get_int('batch_size'), shuffle=False, num_workers=4)

        self.model = ClassificationCNN(num_filters=self.preset.get_list("num_filters"), kernel_size=self.preset.get_int("kernel_size"), hidden_dims=self.preset.get_list('hidden_dims'), dropout=self.preset.get_float('dropout'))
        self.solver = Solver()
        self.solver.set_model(self.model)
        self.train_iterator = iter(self.train_loader)

    def save(self, path):
        self.model.save(str(path / "classification_cnn.model"))

    def step(self, tensorboard_writer, current_iteration):
        try:
            acc, loss = self.solver.step(self.model, self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            acc, loss = self.solver.step(self.model, self.train_iterator)

        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/training", simple_value=loss)]), current_iteration)
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/training", simple_value=acc)]), current_iteration)

        if current_iteration % self.preset.get_int('val_interval') == 0:
            val_acc, val_loss = self.solver.validate(self.model, self.val_loader)
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/val", simple_value=val_loss)]), current_iteration)
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/val", simple_value=val_acc)]), current_iteration)

    def load(self, path):
        self.model = torch.load(str(path / "classification_cnn.model"))
        self.solver.set_model(self.model)
