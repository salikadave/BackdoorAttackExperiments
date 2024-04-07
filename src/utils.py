import torch
import torch.nn.functional as F
import random

import os
import matplotlib.pyplot as plt
import numpy as np


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
    # visualize_image(image, filename='original_image.png')

    image += perturbation
    image *= 255
    image = image.round()
    image /= 255
    image = image.clamp(0, 1)

    # visualize_image(image, filename='perturbed_image.png')
    
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
    # plt.savefig(os.path.join('./samples', filename))
    plt.imshow(new_image)
    plt.savefig(os.path.join('./samples', filename), format='png')
    # plt.show()
