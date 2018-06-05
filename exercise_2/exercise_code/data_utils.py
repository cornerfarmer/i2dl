import pickle as pickle
import numpy as np
import os
from exercise_code.features import *
import cv2


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = np.array(datadict['data'])
        Y = np.array(datadict['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    f = os.path.join(ROOT, 'cifar10_train.p')
    Xtr, Ytr = load_CIFAR_batch(f)
    return Xtr, Ytr


def get_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/'
    X, y = load_CIFAR10(cifar10_dir)

    # Subsample the data
    # Our training set will be the first num_train points from the original
    # training set.
    mask = list(range(num_training))
    X_train = X[mask]
    y_train = y[mask]

    # Our validation set will be num_validation points from the original
    # training set.
    mask = list(range(num_training, num_training + num_validation))
    X_val = X[mask]
    y_val = y[mask]

    # We use a small subset of the training set as our test set.
    mask = list(range(num_training + num_validation, num_training + num_validation + num_test))
    X_test = X[mask]
    y_test = y[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Transpose so that channels come first
    #X_train = X_train.transpose(0, 3, 1, 2).copy()
    #X_val = X_val.transpose(0, 3, 1, 2).copy()
    #X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }

def data_augm(images, labels, factor, scale_min, scale_max):
    new_images = np.zeros([images.shape[0] * factor, images.shape[1], images.shape[2], images.shape[3]])
    new_labels = np.zeros([labels.shape[0] * factor], dtype=np.int)

    index = 0
    next_new_index = 0
    for image in images:
        for i in range(factor):
            new_images[next_new_index] = data_augm_image(np.array(image), scale_min, scale_max)
            new_labels[next_new_index] = labels[index]
            next_new_index += 1
        index += 1
    return new_images, new_labels

def data_augm_image(image, scale_min, scale_max):
    img_size = image.shape[0]
    scale_size = np.random.randint(img_size * scale_min, img_size * scale_max)
    image = cv2.resize(image, dsize=(scale_size, scale_size), interpolation=cv2.INTER_CUBIC)

    top = np.random.randint(0, scale_size - img_size)
    left = np.random.randint(0, scale_size - img_size)
    image = image[top:top + img_size, left:left + img_size, :]
    return image


def get_feature_extracted_CIFAR10_data(num_training=48000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/'
    X, y = load_CIFAR10(cifar10_dir)

    # Subsample the data
    # Our training set will be the first num_train points from the original
    # training set.
    mask = list(range(num_training))
    X_train = X[mask]
    y_train = y[mask]

    # Our validation set will be num_validation points from the original
    # training set.
    mask = list(range(num_training, num_training + num_validation))
    X_val = X[mask]
    y_val = y[mask]

    # We use a small subset of the training set as our test set.
    mask = list(range(num_training + num_validation, num_training + num_validation + num_test))
    X_test = X[mask]
    y_test = y[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    num_color_bins = 10  # Number of bins in the color histogram
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)

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
    X_train = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def scoring_function(x, lin_exp_boundary, doubling_rate):
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x <= lin_exp_boundary)
    exp_region = np.logical_and(x > lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score
