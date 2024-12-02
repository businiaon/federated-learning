#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# fl_main.py
"""
Experiment Module Functionality: This module serves as the main control unit for Federated Learning experiments.
It handles dataset loading, model initialization, configuration of experiment parameters, and invokes the Federated
Learning module (federate_learning.py) to execute the experiment.

Author: Lina Liu
Date: 2024-11-30
"""

import numpy as np
import torch
from data import fetch_dataset, iid_partition_loader, noniid_partition_loader
from models import MLP, CNN, LogisticRegression
from federated_learning import fed_avg
from config import get_config
from util import num_params, plot_federated_averaging


def run_federated_experiment(config, model, client_loader, test_loader, data_type):
    """
    A common method to run the Federated Learning experiment.
    Train the model and save results based on different configurations.
    """
    # Train model
    acc_iid = fed_avg(
        global_model=model,
        client_loaders=client_loader,
        num_clients_per_round=config.num_clients_per_round,
        num_local_epochs=config.num_local_epochs,
        lr=config.learning_rate,
        test_loader=test_loader,
        num_rounds=config.num_rounds
    )
    np.save(f'./{config.save_path}/{config.model_name}_{data_type}_acc.npy', acc_iid)


def create_model(config):
    """
    Create different models based on the configuration.
    """
    if config.model_name == "LogisticRegression":
        model = LogisticRegression(input_dim=config.input_dim, num_classes=config.num_classes)
    elif config.model_name == "MLP":
        model = MLP(input_dim=config.input_dim, hidden_dim=config.hidden_dim, num_classes=config.num_classes)
    elif config.model_name == "CNN":
        model = CNN(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    return model


def run_experiment(config):
    """
    The main method to run the experiment, which loads data and trains the model according to the configuration.
    """
    # Load data based on the configuration

    train_data, test_data = fetch_dataset(config.dataset)

    # Use data loaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)  # Inference bsz=1000
    debug_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size)

    # Prepare client data according to IID or Non-IID data partitioning
    if config.iid:
        client_loader = iid_partition_loader(train_data, bsz=config.batch_size)
    else:
        client_loader = noniid_partition_loader(train_data, bsz=config.batch_size)

    # Create model
    model = create_model(config)

    # Move the model to the specified device (CPU or CUDA)
    model = model.to(config.device)

    # Print model information
    print(model)
    print(f"Total parameters: {num_params(model)}")

    # Execute Federated Learning experiment
    if config.iid:
        data_type = 'iid'
    else:
        data_type = 'noniid'
    run_federated_experiment(config, model, client_loader, test_loader, data_type)


if __name__ == "__main__":
    # Get dynamic configuration
    config = get_config()
    run_experiment(config)

    # Define the x-axis (communication rounds)
    # x_cifar10 = np.arange(1, 101)
    # x_mnist = np.arange(1, 101)
    # x_mnist_non_iid = np.arange(1, 301)

    # CIFAR-10 IID Federated Averaging plot
    # Load experiment results for CIFAR-10 and MNIST
    # acc_rl_iid_m10 = np.load('./save/mnist/acc_rl_iid_m10.npy')
    # acc_rl_iid_m50 = np.load('./save/mnist/acc_rl_iid_m50.npy')
    # acc_mlp_iid_m10 = np.load('./save/mnist/acc_mlp_iid_m10.npy')
    # acc_mlp_iid_m50 = np.load('./save/mnist/acc_mlp_iid_m50.npy')
    # acc_cnn_iid_m10 = np.load('./save/mnist/acc_cnn_iid_m10.npy')
    #
    # acc_rl_noniid_m10 = np.load('./save/mnist/acc_rl_noniid_m10.npy')
    # acc_rl_noniid_m50 = np.load('./save/mnist/acc_rl_noniid_m50.npy')
    # acc_mlp_noniid_m10 = np.load('./save/mnist/acc_mlp_noniid_m10.npy')
    # acc_mlp_noniid_m50 = np.load('./save/mnist/acc_mlp_noniid_m50.npy')
    #
    # plot_federated_averaging(
    #     x_cifar10,
    #     [acc_mlp_iid_m10, acc_mlp_iid_m50, acc_cnn_iid_m10],
    #     ['MLP, $m=10$, $E=1$', 'MLP, $m=50$, $E=1$', 'CNN, $m=10$, $E=5$'],
    #     "Federated Averaging on IID CIFAR-10: Test Accuracy Across $t$ Rounds",
    #     "Test accuracy",
    #     (0.3, 0.8),
    #     "./save/cifar10/federated_averaging_iid_cifar.png"
    # )

    # MNIST IID Federated Averaging plot
    # plot_federated_averaging(
    #     x_mnist,
    #     [acc_rl_iid_m10, acc_rl_iid_m50, acc_mlp_iid_m10, acc_mlp_iid_m50, acc_cnn_iid_m10],
    #     ['Logistic Regression, $m=10$, $E=1$', 'Logistic Regression, $m=50$, $E=1$',
    #      'MLP, $m=10$, $E=1$', 'MLP, $m=50$, $E=1$', 'CNN, $m=10$, $E=5$'],
    #     "Federated Averaging on IID MNIST: Test Accuracy Across $t$ Rounds",
    #     "Test accuracy",
    #     (0.9, 1),
    #     "./save/mnist/federated_averaging_iid_mnist.png"
    # )

    # MNIST Non-IID Federated Averaging plot
    # plot_federated_averaging(
    #     x_mnist_non_iid,
    #     [acc_mlp_noniid_m10[:300], acc_mlp_noniid_m50, acc_rl_noniid_m10[:200], acc_rl_noniid_m50[:100]],
    #     ['MLP, $m=10$, $E=1$', 'MLP, $m=50$, $E=1$', 'Logistic Regression, $m=10$, E=5',
    #      'Logistic Regression, $m=50$, E=5'],
    #     "Federated Averaging on Non-IID MNIST: Test Accuracy Across $t$ Rounds",
    #     "Test accuracy",
    #     (0.8, 1),
    #     "./save/mnist/federated_averaging_non_iid_mnist.png"
    # )


