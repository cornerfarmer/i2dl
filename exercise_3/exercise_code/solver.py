from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
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
        self.optim = self.optimF(model.parameters(), **self.optim_args)

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        self.set_model(model)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        for e in range(num_epochs):
            train_iterator = iter(train_loader)
            for i in range(iter_per_epoch):

                train_acc, train_loss = self.step(model, train_iterator)
                self.train_loss_history.append(train_loss)

                if i % log_nth == 0:
                    print("[Iteration " + str(i) + "/" + str(iter_per_epoch) + "] TRAIN loss: " + str(self.train_loss_history[-1]))

            self.train_acc_history.append(train_acc)
            print("[Epoch " + str(e) + "/" + str(num_epochs) + "] TRAIN acc: " + str(self.train_acc_history[-1]) + "/" + str(self.train_loss_history[-1]))

            val_acc, val_loss = self.validate(model, val_loader)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            print("[Epoch " + str(e) + "/" + str(num_epochs) + "] VAL acc: " + str(self.val_acc_history[-1]) + "/" + str(self.val_loss_history[-1]))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

    def validate(self, model, val_loader):
        correct = 0
        loss_val = 0
        counter = 0
        for batch in val_loader:
            output = model(batch[0])
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == batch[1]).sum().item()
            loss_val += self.loss_func(output, batch[1]).item()
            counter += 1

        return correct / len(val_loader.dataset), loss_val / counter

    def step(self, model, train_iterator):
        batch = next(train_iterator)

        self.optim.zero_grad()
        output = model(batch[0])
        _, predicted = torch.max(output.data, 1)

        loss = self.loss_func(output, batch[1])
        loss.backward()

        self.optim.step()
        return (predicted == batch[1]).sum().item() / len(batch[1]), loss.item()