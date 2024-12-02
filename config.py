#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# config.py
"""
Configuration for Federated Learning and Model Training Experiments.

This script defines the configuration parameters used for setting up and running various
machine learning experiments, including federated learning. It provides options for configuring
basic model parameters, dataset selection, federated learning settings, and data distribution type.

Author: Lina Liu
Date: 2024-11-30
"""

import argparse


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for experiments")

    # Basic Configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to train the model on (CPU or GPU)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")

    # Dataset and Model Configuration
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset to use for training (mnist or cifar10)")
    parser.add_argument("--model_name", type=str, default="MLP", choices=["MLP", "CNN", "LogisticRegression"],
                        help="Model to use (MLP, CNN, or LogisticRegression)")
    parser.add_argument("--input_dim", type=int, default=28 * 28, help="Input dimension for the model (e.g., for MNIST)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for MLP model")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes (e.g., 10 for MNIST or "
                                                                    "CIFAR-10)")

    # Federated Learning Configuration
    parser.add_argument("--num_clients_per_round", type=int, default=10, help="Number of clients per round in "
                                                                              "federated learning")
    parser.add_argument("--num_local_epochs", type=int, default=5, help="Number of local epochs to train per client")
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of federated learning rounds")

    # Data distribution type: IID or non-IID
    parser.add_argument("--iid", type=bool, default=True, help="Whether the data is IID (True) or non-IID (False)")

    # Save path for intermediate results (model checkpoints, logs, etc.)
    parser.add_argument("--save_path", type=str, default='./save',
                        help="Path to save intermediate model training results")

    config = parser.parse_args()

    return config
