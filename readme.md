Backdoor Attack

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

Additional task:

(TBD) Implement the patch replacement backdoor pattern

============================================================================
Project Plan:

Sequence for training:

[-] Make code changes locally for attack crafting - DONE
[-] Generate attack samples
[-] Make code changes locally for training
[-] In train(epoch): Write PERT_SIZE, BD_NUM, epoch, train_acc to train_stats.csv
[-] In test(epoch): Write PERT_SIZE, BD_NUM, test_acc to validation_stats.csv
[-] In test_attack(epoch): Write PERT_SIZE, BD_NUM, test_acc to asr_stats.csv
[-] Plot graphs based on dfs in 3 csv files
    [-] Determine highest ASR when PERT_SIZE constant
    [-] Determine highest ASR when BD_NUM constant
    [-] Determine highest ASR when BD_NUM constant
    [-] Training loss/epoch v/s BD_NUM (for highest ASR)
    [-] Training loss/epoch v/s PERT_SIZE (for highest ASR)
    [-] ASR v/s Poisoning rate (calc from BD_NUM)
    [-] ASR v/s PERT_SIZE
[-] Add args in attacks_crafting.py to accept PERT_SIZE and BD_NUM
[-] Add args in train.py to set path for train_attacks and test_attacks

A. Craft Samples:
    - Keeping PERT_SIZE = 2/255
        - BD_NUM = 250
        - BD_NUM = 500
        - BD_NUM = 750
        - BD_NUM = 1000
        - BD_NUM = 1500
    - Keeping PERT_SIZE = 4/255
        - BD_NUM = 250
        - BD_NUM = 500
        - BD_NUM = 750
        - BD_NUM = 1000
        - BD_NUM = 1500
    - Keeping PERT_SIZE = 6/255
        - BD_NUM = 250
        - BD_NUM = 500
        - BD_NUM = 750
        - BD_NUM = 1000
        - BD_NUM = 1500

B. Train RN-18
