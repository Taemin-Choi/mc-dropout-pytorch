# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import dataset
import net

class RegressionTrainer(object):
    """
    """
    def __init__(self, shape):
        self.shape = shape
        torch.cuda.is_available()
        # print("GPU {}".format(use_gpu))
        # print("GPU count {}".format(torch.cuda.device_count()))
        device_index =torch.cuda.current_device()
        print("Using {}".format(torch.cuda.get_device_name(device_index)))

        # For reproducivility
        torch.manual_seed(777)

        x_train, y_train, x_test = dataset.load_dataset(shape)

        self.X = torch.Tensor(x_train)
        self.Y = torch.Tensor(y_train)
        self.X_test = torch.Tensor(x_test)

        self.model = net.load_model('simple')

    def train(self, learning_rate, epoch):
        # cost criterion
        criterion = nn.MSELoss()

        # Minimize
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Train the model
        self.model.train()
        for step in range(epoch):
            optimizer.zero_grad()
            # Our hypothesis
            hypothesis = self.model(self.X)
            cost = criterion(hypothesis, self.Y)
            cost.backward()
            optimizer.step()

            if step % 100 == 0:
                print(step, cost.data.numpy())

        save_name = './weights/' + str(self.shape) + '.pth' 
        torch.save(self.model.state_dict(), save_name)

    def eval(self):
        # Eval the model
        self.model.eval()
        self.Y_pred = self.model(self.X_test)

    def visualize(self):
        print("heee")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X, self.Y, c='b', label='train')

        # Tensor to Numpy
        X_test = self.X_test.numpy()
        Y_pred = self.Y_pred.detach().numpy()
        ax.scatter(X_test, Y_pred, c='r', label='test')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='curve', type=str)
    args = parser.parse_args()

    regression_trainer = RegressionTrainer(args.shape)
    print('Start training ...')
    regression_trainer.train(0.01, 50000)
    print('Done training.')
    regression_trainer.eval()
    regression_trainer.visualize()

    # python train.py --weights YOLO_small.ckpt --gpu 0