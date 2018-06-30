"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models

class SegmentationNN(nn.Module):

    def __init__(self, img_size, num_classes=23, mode=0):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        self.mode = mode
        self.img_size = img_size
        self.model_ft = models.vgg16(pretrained=True)
        next(iter(self.model_ft.features)).padding = [100, 100]
        self.full_conv = False
        self.transform_to_fully_conv()

        for p in self.model_ft.features.parameters():
            p.requires_grad = False

        for p in self.model_ft.classifier.parameters():
            p.requires_grad = False

        self.conv = nn.Conv2d(4096, num_classes, 1)
        self.convtransp = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)
        self.upsample = nn.Upsample(scale_factor=32)

    def transform_to_fully_conv(self):
        new_classifier = []
        filter_size = 7
        for module in self.model_ft.classifier.children():
            if type(module) is nn.Linear:
                new_classifier.append(self.fc_to_conv([module.in_features / (filter_size ** 2), filter_size, filter_size], module))
                filter_size = 1
            else:
                new_classifier.append(module)
        self.model_ft.classifier = nn.Sequential(*new_classifier[:-1])
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
        if self.mode is 0 or self.mode is 1:
            x = self.model_ft.features(x)
            if not self.full_conv:
                x = x.view(x.size(0), -1)
            x = self.model_ft.classifier(x)
            #print(x.size())
            #print(np.argmax(x.detach().numpy()[0], 0))

        if self.mode is 0 or self.mode is 2:
            x = self.conv(x)
            #x = self.convtransp(x)
            x = self.upsample(x)
            offset = (x.size()[2] - self.img_size) // 2, (x.size()[3] - self.img_size) // 2
            #print(x.size(), orig_x.size(), offset)
            x = x[:, :, offset[0]:offset[0] + self.img_size, offset[1]:offset[1] + self.img_size].contiguous()
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
