import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()
        self.num_models = 15

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.features = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        for i in range(self.num_models):
            features = []
            in_channel = 1
            out_channel = 32
            filter = 4
            dropout = 0.1
            for i in range(4):
                features.append(nn.Conv2d(in_channel, out_channel, filter))
                features.append(nn.ReLU())
                features.append(nn.MaxPool2d(2))
                features.append(nn.Dropout(dropout))

                filter -= 1
                in_channel = out_channel
                out_channel *= 2
                dropout += 0.1

            self.features.append(nn.Sequential(*features))

            classifier = []
            in_features = 6400
            for i in range(2):
                classifier.append(nn.Linear(in_features, 1000))
                classifier.append(nn.ReLU())
                classifier.append(nn.Dropout(dropout))

                dropout += 0.1
                in_features = 1000

            classifier.append(nn.Linear(in_features, 2))

            self.classifiers.append(nn.Sequential(*classifier))
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, img):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        out = torch.zeros([img.size()[0], 30])

        for i in range(self.num_models):
            x = self.features[i](img)
            x = x.view(x.size(0), -1)
            out[:, i*2:i*2+2] = self.classifiers[i](x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
