import sys
sys.path.append('../../')
import TaskPlan
from exercise_code.data_utils import get_CIFAR10_data
import numpy as np
from exercise_code.classifiers.fc_net import FullyConnectedNet
import tensorflow as tf
import pickle
from exercise_code.solver import Solver

class Task(TaskPlan.Task):

    def __init__(self, preset, logger, subtask):
        super().__init__(preset, logger, subtask)

        data = get_CIFAR10_data()
        full_data = {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
        }

        self.net = FullyConnectedNet(self.preset.get_list('hidden_size')[:], weight_scale=self.preset.get_float('weight_scale'), use_batchnorm=self.preset.get_bool('use_batchnorm'), dropout=self.preset.get_float('dropout'), reg=self.preset.get_float('reg'))
        self.solver = Solver(self.net, full_data,
                        num_epochs=50, batch_size=self.preset.get_int('batch_size'),
                        update_rule=self.preset.get_string('update_rule'),
                        optim_config={
                            'learning_rate': self.preset.get_float('learning_rate')
                        },
                        verbose=False, print_every=100000)

    def save(self, path):
        pickle.dump({'fully_connected_net': self.net}, open(str(path / 'fully_connected_net.p'), 'wb'))

    def step(self, tensorboard_writer, current_iteration):
        loss = self.solver.step()
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/training", simple_value=loss)]), current_iteration)

        if current_iteration % self.preset.get_int('val_interval') == 0:
            train_acc, val_acc = self.solver.check_all_accuracies()
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/training", simple_value=train_acc)]), current_iteration)
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/val", simple_value=val_acc)]), current_iteration)

    def load(self, path):
        self.net = pickle.load(open(str(path / 'fully_connected_net.p'), 'rb'))['fully_connected_net']
        self.solver.set_model(self.net)
