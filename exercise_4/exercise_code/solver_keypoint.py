from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import gc

class SolverKeyPoint(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optimF = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def set_model(self, model):
        self.optim = []
        for i in range(15):
            self.optim.append(self.optimF(model.parameters_for_model(i), **self.optim_args))

    def validate(self, model, val_loader):
        model.eval()
        loss_val = 0
        counter = 0
        for batch in val_loader:
            images = batch['image']
            key_pts = batch['keypoints']
            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            output = model(images)

            key_pts[torch.isnan(key_pts)] = output[torch.isnan(key_pts)].detach()
            loss_val += self.loss_func(output, key_pts).item()
            counter += 1

        return loss_val / counter, 1.0 / (2 * (loss_val/len(val_loader)))

    def step(self, model, train_iterators, train_loaders):
        losses = []
        for i in range(len(train_iterators)):
            try:
                batch = next(train_iterators[i])
            except StopIteration:
                train_iterators[i] = iter(train_loaders[i])
                batch = next(train_iterators[i])
                print("restart")

            model.train()

            self.optim[i].zero_grad()

            images = batch['image']
            key_pts = batch['keypoints']
            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            #print(images.size())

            model.selected_model = i
            output = model(images)
            model.selected_model = -1

            #key_pts[torch.isnan(key_pts)] = output[torch.isnan(key_pts)].detach()

            loss = self.loss_func(output, key_pts)
            loss.backward()

            self.optim[i].step()

            losses.append(loss.item())

        loss_sum = np.sum(losses) / 15
        return loss_sum, 1.0 / (2 * loss_sum), losses