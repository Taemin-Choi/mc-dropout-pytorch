# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import dataset
import net


# categories = ['line', 'curve', 'circle']

# # line curve and circle

# # X and Y
# x1 = np.linspace(0, 6, 90)
# x2 = np.linspace(8, 10, 30)

# x_train = np.append(x1, x2)
# y_train = np.sin(x_train) + 0.1

# x_train_reshape = np.reshape(x_train, (-1, 1))
# y_train_reshape = np.reshape(y_train, (-1, 1))
# X = Variable(torch.Tensor(x_train_reshape))
# Y = Variable(torch.Tensor(y_train_reshape))

# # Model
# model = nn.Sequential(nn.Linear(1, 50, bias=True),
#                       nn.Sigmoid(),
#                       nn.Linear(50, 50, bias=True),
#                       nn.Sigmoid(),
#                       nn.Linear(50, 1, bias=True))
# print(model)

# # cost criterion
# criterion = nn.MSELoss()

# # Minimize
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# # Train the model
# model.train()
# for step in range(100001):
#     optimizer.zero_grad()
#     # Our hypothesis
#     hypothesis = model(X)
#     cost = criterion(hypothesis, Y)
#     cost.backward()
#     optimizer.step()

#     if step % 20 == 0:
#         print(step, cost.data.numpy())

# # x_test = [2, 4, 6, 8]
# # y_test = []

# x_test = np.linspace(0, 8, 120)
# x_test_reshape = np.reshape(x_test, (-1, 1))
# X_test = Variable(torch.Tensor(x_test_reshape))

# predicted = model(X_test)
# y_test = predicted.data.numpy()
# print(x_test.shape)
# print(y_test.shape)
# #     y_test.append(output_y)
# # print(y_test)
# # # # x_test = np.array(x_test)
# # # # y_test = np.array(y_test)

# # # print(x_test)
# # # print(y_test)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x_train, y_train, c='b')
# ax.scatter(x_test, y_test, c='r')
# plt.show()

class RegressionTrainer(object):
    """
    """
    def __init__(self, shape):
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

    def 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='line', type=str)
    args = parser.parse_args()

    regression_trainer = RegressionTrainer(args.shape)
    regression_trainer.train(0.01, 1000)
    # print(args.shape)



    # solver = Solver(yolo, pascal)

    # print('Start training ...')
    # solver.train()
    # print('Done training.')

    # python train.py --weights YOLO_small.ckpt --gpu 0