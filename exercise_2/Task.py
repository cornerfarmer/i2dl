import sys
sys.path.append('../../')
import TaskPlan
from exercise_code.data_utils import get_CIFAR10_data, data_augm, extract_features_initial, extract_features_of_images
import numpy as np
from exercise_code.classifiers.fc_net import FullyConnectedNet
import tensorflow as tf
import pickle
from exercise_code.solver import Solver
from io import StringIO
import matplotlib.pyplot as plt

class Task(TaskPlan.Task):

    def __init__(self, preset, logger, subtask):
        super().__init__(preset, logger, subtask)

        self.data = get_CIFAR10_data()
        x_train, y_train = data_augm(self.data['X_train'], self.data['y_train'], 2, self.preset.get_float('scale_min'), self.preset.get_float('scale_max'), self.preset.get_int('translate_max'))
        full_data = {
            'X_train': x_train,
            'y_train': y_train,
            'X_val': self.data['X_val'],
            'y_val': self.data['y_val'],
        }

        if self.preset.get_bool('extract_features'):
            full_data, self.mean_feat, self.std_feat = extract_features_initial(full_data)
            self.logger.log("Extracting features")

        self.net = FullyConnectedNet(self.preset.get_list('hidden_size')[:], input_dim=np.prod(full_data['X_train'].shape[1:]), weight_scale=self.preset.get_float('weight_scale'), use_batchnorm=self.preset.get_bool('use_batchnorm'), dropout=self.preset.get_float('dropout'), reg=self.preset.get_float('reg'))

        if self.preset.get_bool('extract_features'):
            self.net.mean_feat, self.net.std_feat = self.mean_feat, self.std_feat

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
        if self.preset.get_bool('data_augmentation') and current_iteration % int(self.data['X_train'].shape[0] / self.preset.get_int('batch_size')) == 0:
            x_train, y_train = data_augm(self.data['X_train'], self.data['y_train'], 1, self.preset.get_float('scale_min'), self.preset.get_float('scale_max'), self.preset.get_int('translate_max'))
            if self.preset.get_bool('extract_features'):
                x_train = extract_features_of_images(x_train, self.mean_feat, self.std_feat)
                self.logger.log("Extracting features")

            full_data = {
                'X_train': x_train,
                'y_train': y_train,
                'X_val': self.solver.X_val,
                'y_val': self.solver.y_val,
            }


            self.logger.log("Doing data augm")
            self.solver.set_data(full_data)

            #s = StringIO()
            #plt.imsave(s, (x_train[0] - x_train[0].min()) / (x_train[0].max() - x_train[0].min()) , format='png')
            #tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/training", image=tf.Summary.Image(encoded_image_string=s, width=32, height=32))]), current_iteration)

        loss = self.solver.step()
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/training", simple_value=loss)]), current_iteration)

        if current_iteration % self.preset.get_int('val_interval') == 0:
            train_acc, val_acc = self.solver.check_all_accuracies()
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/training", simple_value=train_acc)]), current_iteration)
            tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/val", simple_value=val_acc)]), current_iteration)

    def load(self, path):
        self.net = pickle.load(open(str(path / 'fully_connected_net.p'), 'rb'))['fully_connected_net']
        self.solver.set_model(self.net)
