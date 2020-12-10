# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()

# For reproducivility
torch.manual_seed(777)

# X and Y
x1 = np.linspace(0, 6, 90)
x2 = np.linspace(8, 10, 30)

x_train = np.append(x1, x2)
y_train = np.sin(x_train) + 0.1

x_train_reshape = np.reshape(x_train, (-1, 1))
y_train_reshape = np.reshape(y_train, (-1, 1))
X = Variable(torch.Tensor(x_train_reshape))
Y = Variable(torch.Tensor(y_train_reshape))

# Model
model = nn.Sequential(nn.Linear(1, 50, bias=True),
                      nn.Sigmoid(),
                      nn.Linear(50, 50, bias=True),
                      nn.Sigmoid(),
                      nn.Linear(50, 1, bias=True))
print(model)
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

# cost criterion
criterion = nn.MSELoss()

# Minimize
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
model.train()
for step in range(100001):
    optimizer.zero_grad()
    # Our hypothesis
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, cost.data.numpy())

# x_test = [2, 4, 6, 8]
# y_test = []

x_test = np.linspace(0, 8, 120)
x_test_reshape = np.reshape(x_test, (-1, 1))
X_test = Variable(torch.Tensor(x_test_reshape))

predicted = model(X_test)
y_test = predicted.data.numpy()
print(x_test.shape)
print(y_test.shape)
#     y_test.append(output_y)
# print(y_test)
# # # x_test = np.array(x_test)
# # # y_test = np.array(y_test)

# # print(x_test)
# # print(y_test)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_train, y_train, c='b')
ax.scatter(x_test, y_test, c='r')
plt.show()