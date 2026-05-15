import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm
import wandb

def plot_pruning(inter_device_pruning: bool = False):
    api = wandb.Api()
    entity = "distributed_transformer"
    project = "pruningV2" if inter_device_pruning else "pruningV2_normal"
    probability = 0.0
    data_all = {}
    runs = api.runs(entity + "/" + project)

    data = {}
    for run in tqdm(runs):
        data_single = run.history(samples=10000, keys=["test_loss"])
        if "test_loss" not in data_single.columns:
            continue
        
        # filter out runs that do not match the desired probability
        if abs(run.config["model"]["partial_layer_dropout_prob"] - probability) > 0.01:
            continue
        dataset = run.config["dataset"]["name"]
        
        if dataset not in data:
            data[dataset] = {}
        
        # Determine amount of communication based on pruning type
        num_communication = None
        if not inter_device_pruning:
            num_communication = run.config["model"]["num_features_attention"] / 8
        elif inter_device_pruning:
            if "pruning_step" not in run.config:
                continue
            num_communication = 16 * (1 - run.config["pruning_step"] * run.config["per_step_pruning_ratio"])
        
        # add data point
        if num_communication not in data[dataset]:
            data[dataset][num_communication] = [data_single["test_loss"].iloc[0]]
        else:
            data[dataset][num_communication].append(data_single["test_loss"].iloc[0])


    def plot_shaded_area(data, label, filename=None):
        nums_communications = sorted(data.keys())
        accuracies = [data[num_com] for num_com in nums_communications]
        means = [np.mean(acc) for acc in accuracies]
        mins = [np.min(acc) for acc in accuracies]
        maxs = [np.max(acc) for acc in accuracies]

        if filename is not None:
            df = pd.DataFrame({
                "message_loss_prob": [num_com for num_com in nums_communications],
                "mean_accuracy": means,
                "min_accuracy": mins,
                "max_accuracy": maxs
            })

            df.to_csv(f"/home/alex/Documents/Paper/papers-dsme-nes/distributed_neural_network_inference/plot_data/{filename}.csv", index=False)

        plt.plot(message_loss_probs, means, label=label)
        plt.fill_between(message_loss_probs, np.array(mins), np.array(maxs), alpha=0.3)

    for dataset in data:
        plt.figure(figsize=(10, 6))

        for k in data_all:
            print(data_all[k][dataset])
            plot_shaded_area(data_all[k][dataset], k, "normal_pruning_vs_accuracy" if not inter_device_pruning else "inter_device_pruning_vs_accuracy")
        
        
        plt.xlabel("Num communication")
        plt.ylabel("Loss")
        plt.title(f"Dataset: {dataset}")
        plt.legend()
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    # plot_pruning(inter_device_pruning=False)
    plot_pruning(inter_device_pruning=True)