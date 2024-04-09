import os
import matplotlib.pyplot  as plt
import argparse
import csv
import pandas as pd

def process_stat_value(line):
    return float(line.split(": ")[1].strip())

def extract_stats(file_path):
    attack_success_rate = list()
    train_acc = list()
    test_acc = list()
    with open(file_path) as file:
        content = file.readlines()
    
    for line in content:
        if "Train" in line:
            train_acc.append(process_stat_value(line))
        elif "Test" in line:
            test_acc.append(process_stat_value(line))
        elif "Attack" in line:
            attack_success_rate.append(process_stat_value(line))
    
    # print(train_acc)

    # return train_acc[:60], test_acc[:60], attack_success_rate[:60]
    return train_acc, test_acc, attack_success_rate
    
def prepare_clean_stats_file():
    clean_stats_file = 'clean_stats.csv'
    if not os.path.exists(clean_stats_file):
        with open(stats_file, 'w') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            writer.writerow(("Epoch", "Train_Acc", "Test_Acc"))
    
    epochs_col = [i for i in range(70)]

    train_acc, test_acc, _ = extract_stats(f'training_logs/{args.filename}')
    
    rows = zip(epochs_col, train_acc, test_acc)
    with open(clean_stats_file, "a") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    stats_file = 'stats.csv'
    parser = argparse.ArgumentParser(description='Crafting statistics for training')
    parser.add_argument("--pert_size", default=1, type=int, help="Refer to mapping for PERT_SIZE = Value in [0, 1]")
    parser.add_argument("--sample_count", default=250, help="BD_NUM = Value in [0, 250, 500, 750, 1000, 1500]")
    parser.add_argument("--filename", default='clean_training.txt', help="Format like sample_logs.txt")
    parser.add_argument("--visualize", default=False, help="Generate graphs for stats.csv")
    args = parser.parse_args()

    if args.filename == 'clean_training.txt':
        prepare_clean_stats_file()
        os._exit(0)

    if args.visualize:
        pass
    
    train_acc, test_acc, attack_success_rate = extract_stats(f'training_logs/{args.filename}')

    if not os.path.exists(stats_file):
        with open(stats_file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            # writer.writerow(("Epoch", "Pert_Size", "BD_Num", "Train_Acc", "Test_Acc", "ASR"))
            writer.writerow(("Epoch", "Train_Acc", "Test_Acc"))

    epochs_col = [i for i in range(60)]
    pert_size_col = [args.pert_size] * 60
    bd_num_col = [args.sample_count] * 60

    rows = zip(epochs_col, pert_size_col, bd_num_col, train_acc, test_acc, attack_success_rate)

    with open(stats_file, "a") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    # plt.plot(epochs, train_acc, label='Training Accuracy')
    # plt.plot(epochs, test_acc, label='Validation Accuracy')
    # plt.plot(epochs, attack_success_rate, label='Attack Success Rate')
    # plt.show()
    # plt.savefig(os.path.join('./samples', 'training_stats.png'))