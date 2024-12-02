#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# utils.py

"""
Utility Functions Module Functionality: This module provides general-purpose utility functions used throughout the
experiment, such as loading datasets, preparing client data, and creating models.

Author: Lina Liu
Date: 2024-11-30
"""

import matplotlib.pyplot as plt
import IPython
import numpy as np

# CIFAR-10 labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def view_10(img, label):
    """ view 10 labelled examples from tensor"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(label[i].cpu().numpy())
        ax.imshow(img[i][0], cmap="gray")
    IPython.display.display(fig)
    plt.close(fig)


# Display CIFAR-10 images and labels
def imshow_cifar10(img, labels):
    """ Display images and labels from the CIFAR-10 dataset """

    # Denormalize the image (restore to original image range)
    img = img / 2 + 0.5  # Denormalize (restores the image to its original range)
    img = img.numpy()  # Convert the tensor to a NumPy array
    img = np.transpose(img,
                       (0, 2, 3, 1))  # Convert to [N, H, W, C] format (N=batch size, H=height, W=width, C=channels)

    # Create a 2x5 grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        ax.axis("off")  # Turn off the axis
        ax.set_title(f"{classes[labels[i]]}")  # Set the title as the corresponding label
        ax.imshow(img[i])  # Display the image

    plt.show()


# Function to plot the results
def plot_federated_averaging(x, acc_list, labels, title, ylabel, yaxis_limits, save_path):
    """
    A function to plot Federated Averaging results.

    Parameters:
        x (np.array): Array of communication rounds.
        acc_list (list): List of accuracy arrays to plot.
        labels (list): List of labels for each accuracy array.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        yaxis_limits (tuple): Limits for the y-axis (min, max).
        save_path (str): Path to save the generated plot.
    """
    plt.figure(figsize=(8, 6))

    plt.title(title)
    plt.xlabel("Communication rounds $t$")
    plt.ylabel(ylabel)
    plt.axis([x[0], x[-1], yaxis_limits[0], yaxis_limits[1]])

    # Add horizontal lines (optional)
    plt.axhline(y=yaxis_limits[0], color='r', linestyle='dashed')
    plt.axhline(y=yaxis_limits[1], color='b', linestyle='dashed')

    # Plot each accuracy array
    for acc, label in zip(acc_list, labels):
        plt.plot(x, acc, label=label)

    plt.legend()

    # Save the figure as a PNG image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def num_params(model):
    """ Calculate the number of parameters in the model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)