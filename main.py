#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# main.py
"""
Direct Model Training Script: This script is the entry point for running model training on a specified dataset
without Federated Learning. It loads the dataset, initializes the model, trains it, and performs evaluation
on the validation/test set. This script is typically used for training and evaluating single models such as MLP or CNN
on datasets like MNIST or CIFAR-10.

Author: Lina Liu
Date: 2024-11-30
"""

import torch
from models import MLP, CNN, LogisticRegression, num_params
from data import load_data
from train import train, test
from config import get_config


def main():
    # get parameters
    config = get_config()

    # set random seed
    torch.manual_seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed(config.seed)

    # loading data
    train_loader, test_loader = load_data(batch_size=config.batch_size)

    # model selection
    if config.model_name == "LogisticRegression":
        model = LogisticRegression(input_dim=config.input_dim, num_classes=config.num_classes)
    elif config.model_name == "MLP":
        model = MLP(input_dim=config.input_dim, hidden_dim=config.hidden_dim, num_classes=config.num_classes)
    elif config.model_name == "CNN":
        model = CNN(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    model = model.to(config.device)

    # print model information
    print(model)
    print(f"Total parameters: {num_params(model)}")

    # loss function & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # train & test
    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, config.device)
        test_loss, test_accuracy = test(model, test_loader, criterion, config.device)

        print(f"Epoch {epoch}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
