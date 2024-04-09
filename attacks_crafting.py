from __future__ import absolute_import
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import json
import numpy as np

from src.utils import pattern_craft, add_backdoor

parser = argparse.ArgumentParser(description='PyTorch Backdoor Attack Crafting')
parser.add_argument("--pert_size", default=1, type=int, help="Refer to mapping for PERT_SIZE = Value in [0, 1]")
parser.add_argument("--sample_count", default=250, help="BD_NUM = Value in [0, 250, 500, 750, 1000, 1500]")
# parser.add_argument("--pattern_type", default='chessboard', help="chessboard")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Load raw data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create the backdoor pattern (TO DO)
perturbation = pattern_craft(trainset.__getitem__(0)[0].size(), config['PATTERN_TYPE'], (int(args.pert_size) / 255))

# Crafting training backdoor images
train_images_attacks = None
train_labels_attacks = None
ind_train = [i for i, label in enumerate(trainset.targets) if label==config["SC"]]
ind_train = np.random.choice(ind_train, int(args.sample_count), False)
for i in ind_train:
    if train_images_attacks is not None:
        train_images_attacks = torch.cat([train_images_attacks, add_backdoor(trainset.__getitem__(i)[0], perturbation).unsqueeze(0)], dim=0) # (TO DO)
        train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([config["TC"]], dtype=torch.long)], dim=0)
    else:
        train_images_attacks = add_backdoor(trainset.__getitem__(i)[0], perturbation).unsqueeze(0)
        train_labels_attacks = torch.tensor([config["TC"]], dtype=torch.long)

# Crafting test backdoor images
test_images_attacks = None
test_labels_attacks = None
ind_test = [i for i, label in enumerate(testset.targets) if label==config["SC"]]
for i in ind_test:
    if test_images_attacks is not None:
        test_images_attacks = torch.cat([test_images_attacks, add_backdoor(testset.__getitem__(i)[0], perturbation).unsqueeze(0)], dim=0)
        test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([config["TC"]], dtype=torch.long)], dim=0)
    else:
        test_images_attacks = add_backdoor(testset.__getitem__(i)[0], perturbation).unsqueeze(0)
        test_labels_attacks = torch.tensor([config["TC"]], dtype=torch.long)

# Create attack dir and save attack images
if not os.path.isdir('attacks'):
    os.mkdir('attacks')
train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}
# torch.save(train_attacks, './attacks/train_attacks')
# torch.save(test_attacks, './attacks/test_attacks')
# torch.save(ind_train, './attacks/ind_train')

torch.save(train_attacks, f'./attacks/train/train_attacks_{args.sample_count}_{args.pert_size}')
torch.save(test_attacks, f'./attacks/test/test_attacks_{args.sample_count}_{args.pert_size}')
torch.save(ind_train, f'./attacks/ind/ind_train_{args.sample_count}_{args.pert_size}')
