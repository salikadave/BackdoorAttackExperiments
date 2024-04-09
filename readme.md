## Backdoor Attack

Goal: Implement backdoor attack

Dataset: CIFAR-10

Model Architecture: ResNet-18

Introduction to backdoor attacks:

BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain (Gu 2017)

url: https://arxiv.org/abs/1708.06733#

Overall procedure:

1. Load in the CIFAR-10 dataset

2. Create a backdoor pattern

3. Create backdoor training samples and backdoor test samples by embedding the backdoor pattern into clean samples.

4. Insert the backdoor training samples into the training set (i.e. poison it) say about 3000 clean (unpoisoned) training samples per class.

5. Load in the model architecture (ResNet-18)

6. Perform training (backpropagation)

7. Evaluate attack performance, that is, accuracy on clean test samples and the attack success rate

8. Report such performance versus the poisoning rate (0, 250, 500, ..,1500 total poisoned images, i.e. up to 5% if 30k clean training samples are used across 10 classes). Also, vary the size of the perturbation (using either L2 or L1 norms) to see the effect of this on the performance for a fixed poisoning rate. 

TO DO: [Follow this first then other parameters in 8. above]

1. Implement a backdoor pattern

2. Poison the training set

3. Modify the configurations: poisoning rate (250 images to 1500 images) OR perturbation size (from 0 to 1)

Report:

1. Attack success rate and clean test accuracy for basic configuration

2. Attack success rate versus poisoning rate or perturbation size
