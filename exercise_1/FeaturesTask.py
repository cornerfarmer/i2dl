import sys
sys.path.append('../../')
import TaskPlan
from exercise_code.data_utils import load_CIFAR10
import numpy as np
from exercise_code.classifiers.neural_net import TwoLayerNet
import tensorflow as tf
import pickle
from exercise_code.features import *

class FeaturesTask(TaskPlan.Task):

    def __init__(self, preset, logger, subtask):
        super().__init__(preset, logger, subtask)
        hidden_size = self.preset.get_int('hidden_size')
        num_classes = 10

        X, y = load_CIFAR10('datasets/')
        # Split the data into train, val, and test sets. In addition we will
        # create a small development set as a subset of the data set;
        # we can use this for development so our code runs faster.
        num_training = 48000
        num_validation = 1000
        num_test = 1000

        # Our training set will be the first num_train points from the original
        # training set.
        mask = range(num_training)
        self.X_train = X[mask]
        self.y_train = y[mask]

        # Our validation set will be num_validation points from the original
        # training set.
        mask = range(num_training, num_training + num_validation)
        self.X_val = X[mask]
        self.y_val = y[mask]

        # We use a small subset of the training set as our test set.
        mask = range(num_training + num_validation, num_training + num_validation + num_test)
        self.X_test = X[mask]
        self.y_test = y[mask]

        num_color_bins = 10  # Number of bins in the color histogram
        feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
        X_train_feats = extract_features(self.X_train, feature_fns, verbose=True)
        X_val_feats = extract_features(self.X_val, feature_fns)
        X_test_feats = extract_features(self.X_test, feature_fns)

        # Preprocessing: Subtract the mean feature
        mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
        X_train_feats -= mean_feat
        X_val_feats -= mean_feat
        X_test_feats -= mean_feat

        # Preprocessing: Divide by standard deviation. This ensures that each feature
        # has roughly the same scale.
        std_feat = np.std(X_train_feats, axis=0, keepdims=True)
        X_train_feats /= std_feat
        X_val_feats /= std_feat
        X_test_feats /= std_feat

        # Preprocessing: Add a bias dimension
        self.X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
        self.X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
        self.X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

        self.net = TwoLayerNet(self.X_train_feats.shape[1], hidden_size, num_classes)

    def save(self, path):
        pickle.dump({'feature_neural_net': self.net}, open(str(path / 'feature_neural_net.p'), 'wb'))

    def step(self, tensorboard_writer, current_iteration):
        loss, acc = self.net.step(self.X_train_feats, self.y_train, learning_rate=self.preset.get_float('learning_rate'), reg=self.preset.get_float('reg'), momentum=self.preset.get_float('momentum'), dropout=self.preset.get_float('dropout'), batch_size=self.preset.get_int('batch_size'))
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss/training", simple_value=loss)]), current_iteration)
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/training", simple_value=acc)]),
                                       current_iteration)

        y_val_pred = self.net.predict(self.X_val_feats)
        tensorboard_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy/val", simple_value=np.mean(self.y_val == y_val_pred))]), current_iteration)

    def load(self, path):
        self.net = pickle.load(open(str(path / 'feature_neural_net.p'), 'rb'))['feature_neural_net']
