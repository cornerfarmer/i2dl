"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        self.model_ft = models.vgg16(pretrained=False)
        self.full_conv = False

    def transform(self):
        new_classifier = []
        filter_size = 7
        for module in self.model_ft.classifier.children():
            if type(module) is nn.Linear:
                new_classifier.append(self.fc_to_conv([module.in_features / (filter_size ** 2), filter_size, filter_size], module))
                filter_size = 1
            else:
                new_classifier.append(module)
        self.model_ft.classifier = nn.Sequential(*new_classifier)
        self.full_conv = True

        #self.conv = nn.Conv2d(512, num_classes, 1)
        #self.conv = self.fc_to_conv([512, 7, 7], self.model_ft.classifier.modules())
        #self.upsample = nn.Upsample([224, 224])#           nn.Linear(self.model_ft.fc.in_features, num_classes)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def fc_to_conv(self, input_dim, fc):
        conv = nn.Conv2d(input_dim[0], fc.out_features, (input_dim[1], input_dim[2]))
        fc_iter = iter(fc.parameters())
        for param in conv.parameters():
            fc_param = next(fc_iter)
            param.data = fc_param.data.view(param.size())
        return conv


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        x = self.model_ft.features(x)
        if not self.full_conv:
            x = x.view(x.size(0), -1)
        x = self.model_ft.classifier(x)
        return x

        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)

        x = self.model_ft.avgpool(x)

        x = self.conv(x)
        x = self.upsample(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
