# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def load_model(deep='simple'):
    if deep == 'simple':
        # Model
        model = nn.Sequential(nn.Linear(1, 100, bias=True),
                              nn.LeakyReLU(0.1),
                              nn.Dropout(0.3),
                              nn.Linear(100, 100, bias=True),
                              nn.LeakyReLU(0.1),
                              nn.Dropout(0.3),
                              nn.Linear(100, 1, bias=True),
                              )

        return model