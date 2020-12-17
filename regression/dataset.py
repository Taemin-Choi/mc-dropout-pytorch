# -*- coding: utf-8 -*-
import numpy as np

def load_dataset(shape):
    if shape == 'line':
        x1 = np.linspace(0, 6, 90)
        x2 = np.linspace(8, 10, 30)  
        x = np.append(x1, x2)
        y = np.sin(x) + 0.1
        x_train = np.reshape(x, (-1, 1))
        y_train = np.reshape(y, (-1, 1))

        x_test = np.reshape(np.linspace(0, 10, 120), (-1, 1))

        return (x_train, y_train, x_test)

    elif shape == 'curve':
        x1 = np.linspace(0, (1 * np.pi), 100)
        x2 = np.linspace((3 / 2 * np.pi), (5 / 2 * np.pi), 100)  
        x = np.append(x1, x2)
        y = np.sin(x)
        x_train = np.reshape(x, (-1, 1))
        y_train = np.reshape(y, (-1, 1))

        x_test = np.reshape(np.linspace(0, (5 / 2 * np.pi), 250), (-1, 1))

        return (x_train, y_train, x_test)

    elif shape == 'circle':
        return 1
