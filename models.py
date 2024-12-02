#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# models.py
"""Model Definition for Federated Learning: This module defines the models that will be used in the federated
learning experiments. Currently, it includes Logistic Regression, MLP and CNN model architectures, with options to
easily extend for other models.

Author: Lina Liu
Date: 2024-11-30
"""

import torch.nn as nn


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) Model"""
    def __init__(self, input_dim=28 * 28, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入张量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """Convolutional Neural Network (CNN) Model"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LogisticRegression(nn.Module):
    """Logistic Regression Model"""
    def __init__(self, input_dim=28 * 28, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output
