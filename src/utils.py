import torch
import torch.nn.functional as F
import random

import os
import matplotlib.pyplot as plt
import numpy as np
import copy as cp


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pattern_craft(im_size, pattern_type, perturbation_size):
    ## TO DO ##
    """
    Add a chessboard pattern to serve as the backdoor trigger.

    (0, 0) (0, 1) (0, 2)
    (1, 0) (1, 1) (1, 2)
    (2, 0) (2, 1) (2, 2)

    X X X      1 X 1
    X X X =>   X 1 X
    X X X      1 X 1
    """
    perturbation = torch.zeros(im_size)
    for i in range(im_size[1]):
        for j in range(im_size[2]):
            if (i + j) % 2 == 0:
                perturbation[:, i, j] = torch.ones(im_size[0])
    perturbation *= perturbation_size
    return perturbation
    # pass


def add_backdoor(image, perturbation):
    ## TO DO ##

    # here is where we will define the backdoor pattern. Let's say the pattern is to add a white square to the bottom right corner of the image. Then the following will be executed:
    '''    
    poisoned_image = image.numpy().copy()
    poisoned_image[:, -5:, -5:] = 1.0  # Adding a white square to the bottom right
    return torch.tensor(poisoned_image), target_label
    '''
    visualize_image(image, filename='original_image.png')
    visualize_perturbation(perturbation)

    image += perturbation
    image *= 255
    image = image.round()
    image /= 255
    image = image.clamp(0, 1)

    visualize_image(image, filename='perturbed_image.png')
    
    return image
    # pass

def visualize_perturbation(pattern):
    """
    Visualize the perturbation.
    """
    pattern = pattern.numpy()
    pattern = np.transpose(pattern, [1, 2, 0])
    plt.imshow(pattern)
    plt.show()
    plt.savefig(os.path.join('.', 'chessdoor_pattern.png'))

def visualize_image(image, filename):
    new_image = image.numpy()
    new_image = np.transpose(new_image, [1, 2, 0])
    plt.imshow(new_image)
    plt.savefig(os.path.join('.', filename), format='png')
    # plt.show()

def pert_est_class_pair(source, target, model, images, labels, pi=0.9, lr=1e-4, NSTEP=100, verbose=False):
    '''
    :param source: souce class
    :param target: target class
    :param model: model to be insected
    :param images: batch of images for perturbation estimation
    :param labels: the target labels
    :param pi: the target misclassification fraction (default is 0.9)
    :param lr: learning rate (default is 1e-4)
    :param NSTEP: number of steps to terminate (default is 100)
    :param verbose: set True to plot details
    :return:
    '''

    if verbose:
        print("Perturbation estimation for class pair (s, t)".format(source, target))

    # Initialize perturbation
    pert = torch.zeros_like(images[0]).to(device)
    pert.requires_grad = True

    for iter_idx in range(NSTEP):

        # Optimizer: SGD
        optimizer = torch.optim.SGD([pert], lr=lr, momentum=0)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Get the loss
        images_perturbed = torch.clamp(images + pert, min=0, max=1)
        outputs = model(images_perturbed)
        loss = criterion(outputs, labels)

        # Update perturbation
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Compute misclassification fraction rho
        misclassification = 0
        with torch.no_grad():
            images_perturbed = torch.clamp(images + pert, min=0, max=1)
            outputs = model(images_perturbed)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            rho = misclassification / len(labels)

        if verbose:
            print("current misclassification: {}; perturbation norm: {}".format(rho, torch.norm(pert).detach().cpu().numpy()))

        # Stopping criteria
        if rho > pi:
            break

    return pert