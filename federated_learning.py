#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# federate_learning.py
"""
Federated Learning Module Functionality: This module implements core functionalities related to Federated
Learning, including client-side training, model validation, model parameter aggregation, and execution of Federated
Averaging (FedAvg) experiments.

Author: Lina Liu
Date: 2024-11-30
"""


import copy
import numpy as np
import torch
from config import get_config

config = get_config()
device = config.device


def validate(model, test_loader):
    """
    Validation function
    """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.size(0)
    return correct / total


def train_client(client_loader, global_model, num_local_epochs, lr, criterion):
    """
    Train a client's model.
    """
    local_model = copy.deepcopy(global_model).to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    local_model.train()
    for _ in range(num_local_epochs):
        for x, y in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(x), y)
            loss.backward()
            optimizer.step()
    return local_model


def aggregate_models(global_model, client_models, num_clients):
    """
    Aggregate client models using the Federated Averaging (FedAvg) algorithm.
    """
    avg_state = None
    for model in client_models:
        if avg_state is None:
            avg_state = {k: v / num_clients for k, v in model.state_dict().items()}
        else:
            for k in avg_state:
                avg_state[k] += model.state_dict()[k] / num_clients
    global_model.load_state_dict(avg_state)
    return global_model


def fed_avg(global_model, client_loaders, num_clients_per_round, num_local_epochs, lr, test_loader, num_rounds,
            criterion):
    """
    The core function of Federated Learning involves conducting multiple training rounds and updating the global
    model at the end of each round.
    """
    accuracies = []
    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")

        # Randomly select clients.
        sampled_clients = np.random.choice(len(client_loaders), num_clients_per_round, replace=False)

        # Train the model for each selected client.
        client_models = [
            train_client(client_loaders[i], global_model, num_local_epochs, lr, criterion)
            for i in sampled_clients
        ]

        # Aggregate client models.
        global_model = aggregate_models(global_model, client_models, num_clients_per_round)

        # Perform validation after each training round.
        acc = validate(global_model, test_loader)
        print(f"Validation accuracy: {acc:.4f}")
        accuracies.append(acc)

    return accuracies
