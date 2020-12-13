# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def load_model(deep):
    if deep == 'simple':
        # Model
        model = nn.Sequential(nn.Linear(1, 50, bias=True),
                              nn.Sigmoid(),
                              nn.Dropout(0.3),
                              nn.Linear(50, 50, bias=True),
                              nn.Sigmoid(),
                              nn.Dropout(0.3),
                              nn.Linear(50, 1, bias=True),
                              )

        return model

    elif deep == 'curve':
        return 1
    elif deep == 'circle':
        return 1
