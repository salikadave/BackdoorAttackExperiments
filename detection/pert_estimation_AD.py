from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import copy as cp
import numpy as np
import time

from src.resnet import ResNet18
from src.utils import pert_est_class_pair

print(os.getcwd())

parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
parser.add_argument("--nc", default=10, type=int, help="")
parser.add_argument("--ni", default=10, type=int, help="")
parser.add_argument("--pi", default=0.9, type=float, help="")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time.time()
random.seed()

# Detection parameters
# NC = 10     # Number of classes
NC = args.nc     # Number of classes
# NI = 2     # Number of images per class used for detection
NI = args.ni     # Number of images per class used for detection
# PI = 0.5
PI = args.pi

print(NC, NI, PI)
sys.exit()

# Load model to be inspected
model = ResNet18()
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
# model.load_state_dict(torch.load('./model/model.pth'))
model.load_state_dict(torch.load('detection/models/model_contam_2_500.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

# Create saving path for results
if not os.path.isdir('pert_estimated'):
    os.mkdir('pert_estimated')

# Load clean images for detection
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Perform pattern estimation for each class pair
for s in range(NC):
    # Get the subset of clean images (from the testset) for detection
    detectionset = cp.copy(testset)
    ind = [i for i, label in enumerate(detectionset.targets) if label == s]
    ind = np.random.choice(ind, NI, False)
    detectionset.data = detectionset.data[ind]
    for t in range(NC):
        # skip the case where s = t
        if s == t:
            continue
        # Set the labels of the images used for detection as t
        detectionset.targets = [t] * len(detectionset.data)
        # Create dataloader
        detectionsetloader = torch.utils.data.DataLoader(detectionset, batch_size=NI, shuffle=False, num_workers=1)
        batch_idx, (images, labels) = list(enumerate(detectionsetloader))[0]
        images, labels = images.to(device), labels.to(device)
        # CORE STEP: perturbation esitmation for (s, t) pair
        pert = pert_est_class_pair(source=s, target=t, model=model, images=images, labels=labels, pi=PI, lr=1e-4)

        torch.save(pert.detach().cpu(), './pert_estimated/pert_estimated_2_500/pert_{}_{}'.format(s, t))

print("--- %s seconds ---" % (time.time() - start_time))
torch.save((time.time() - start_time), './pert_estimated/pert_estimated_2_500/time')