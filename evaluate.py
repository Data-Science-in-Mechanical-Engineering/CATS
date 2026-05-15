import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm
import wandb


def plot_foundational_model_message_loss():
    api = wandb.Api()
    project = "ett_messageloss"
    entity = "distributed_transformer"
    runs = api.runs(entity + "/" + project)
    pruning_step = 0

    data_all = {}
    for run in runs:

        data_test_loss = run.history(samples=10000, keys=["test_loss"])  # , keys=["test_loss", "test_loss_zero_shot", "message_loss"])
        data_test_zero_shot = run.history(samples=10000, keys=["test_loss_zero_shot"])
        print("h")
        print(run.config["dataset"]["name"])
        if "test_loss" not in data_test_loss.columns or "test_loss_zero_shot" not in data_test_zero_shot.columns:
            continue
        if run.config["pruning_step"] != pruning_step:
            continue
        dataset = run.config["dataset"]["name"] + run.config["dataset"]["variant"]
        do_finetuning = run.config["do_finetuning"]
        partial_layer_dropout_prob = run.config["model"]["partial_layer_dropout_prob"]
        if dataset not in data_all:
            data_all[dataset] = {}
        if do_finetuning not in data_all[dataset]:
            data_all[dataset][do_finetuning] = {}
        if partial_layer_dropout_prob not in data_all[dataset][do_finetuning]:
            data_all[dataset][do_finetuning][partial_layer_dropout_prob] = {"test_loss_zero_shot": [], "test_loss": []}
        
        data_all[dataset][do_finetuning][partial_layer_dropout_prob]["test_loss"].append(data_test_loss["test_loss"][0])
        data_all[dataset][do_finetuning][partial_layer_dropout_prob]["test_loss_zero_shot"].append(data_test_zero_shot["test_loss_zero_shot"][0])

    def plot_shaded_area(ax, data, label, loss_name, filename=None):
        partial_layer_dropout_prob = np.array(list(data.keys()))
        partial_layer_dropout_prob = np.sort(partial_layer_dropout_prob)
        loss = np.array([data[p][loss_name] for p in partial_layer_dropout_prob])
        print(loss)
        means = np.mean(loss, axis=1)
        mins = np.min(loss, axis=1)
        maxs = np.max(loss, axis=1)
        ax.plot(partial_layer_dropout_prob, means, label=label)
        ax.fill_between(partial_layer_dropout_prob, np.array(mins), np.array(maxs), alpha=0.3)

        # if filename is not None:
        #     df = pd.DataFrame({
        #         "message_loss_prob": [p * 100 for p in message_loss_probs],
        #         "mean_accuracy": means,
        #         "min_accuracy": mins,
        #         "max_accuracy": maxs
        #     })

            # df.to_csv(f"/home/alex/Documents/009_Paper/papers-dsme-nes/federated_learning/images/{filename}.csv", index=False)


    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            for loss_name, ax in zip(["test_loss_zero_shot", "test_loss"], axes):
                plot_shaded_area(ax, data_all[dataset][True], "Foundational Model", loss_name, f"{dataset}_finetuning_{loss_name}")
                plot_shaded_area(ax, data_all[dataset][False], "Random init", loss_name, f"{dataset}_zero_shot_{loss_name}")
                ax.set_xlabel("Message loss probability")
                ax.set_ylabel("Test Loss")
                ax.set_title(f"Dataset: {dataset} - {loss_name.replace('_', ' ')}")
                ax.legend()    
                ax.grid(True)

            plt.show()
        except:
            print(f"Error plotting dataset {dataset}. Skipping...")


def plot_foundational_model_pruning():
    api = wandb.Api()
    project = "ett_pruning"
    entity = "distributed_transformer"
    runs = api.runs(entity + "/" + project)

    data_all = {}
    for run in runs:

        data_test_loss = run.history(samples=10000, keys=["test_loss"])  # , keys=["test_loss", "test_loss_zero_shot", "message_loss"])
        data_test_zero_shot = run.history(samples=10000, keys=["test_loss_zero_shot"])
        print("h")
        print(run.config["dataset"]["name"])
        if "test_loss" not in data_test_loss.columns or "test_loss_zero_shot" not in data_test_zero_shot.columns:
            continue
        dataset = run.config["dataset"]["name"] + run.config["dataset"]["variant"]
        do_finetuning = run.config["do_finetuning"]
        pruning_step = run.config["pruning_step"]
        print(dataset)
        if dataset not in data_all:
            data_all[dataset] = {}
        if do_finetuning not in data_all[dataset]:
            data_all[dataset][do_finetuning] = {}
        if pruning_step not in data_all[dataset][do_finetuning]:
            data_all[dataset][do_finetuning][pruning_step] = {"test_loss_zero_shot": [], "test_loss": []}
        
        data_all[dataset][do_finetuning][pruning_step]["test_loss"].append(data_test_loss["test_loss"][0])
        data_all[dataset][do_finetuning][pruning_step]["test_loss_zero_shot"].append(data_test_zero_shot["test_loss_zero_shot"][0])

    def plot_shaded_area(ax, data, label, loss_name, filename=None):
        print(data)
        pruning_step = np.array(list(data.keys()))
        pruning_step = np.sort(pruning_step)
        loss = np.array([[data[p][loss_name][0]] for p in pruning_step])
        print(loss)
        means = np.mean(loss, axis=1)
        mins = np.min(loss, axis=1)
        maxs = np.max(loss, axis=1)
        ax.plot(pruning_step * 20, means, label=label)
        ax.fill_between(pruning_step * 20, np.array(mins), np.array(maxs), alpha=0.3)

        # if filename is not None:
        #     df = pd.DataFrame({
        #         "message_loss_prob": [p * 100 for p in message_loss_probs],
        #         "mean_accuracy": means,
        #         "min_accuracy": mins,
        #         "max_accuracy": maxs
        #     })

            # df.to_csv(f"/home/alex/Documents/009_Paper/papers-dsme-nes/federated_learning/images/{filename}.csv", index=False)


    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            for loss_name, ax in zip(["test_loss_zero_shot", "test_loss"], axes):
                plot_shaded_area(ax, data_all[dataset][True], "Foundational Model", loss_name, f"{dataset}_finetuning_{loss_name}")
                plot_shaded_area(ax, data_all[dataset][False], "Random init", loss_name, f"{dataset}_zero_shot_{loss_name}")
                ax.set_xlabel("Pruning Ratio (%)")
                ax.set_ylabel("Test Loss")
                ax.set_title(f"Dataset: {dataset} - {loss_name.replace('_', ' ')}")
                ax.legend()    
                ax.grid(True)

            plt.show()
        except:
            print(f"Error plotting dataset {dataset}. Skipping...")


def plot_message_loss():
    api = wandb.Api()
    project = "ts_pruning"
    entity = "distributed_transformer"
    pruning_step = 4
    runs = api.runs(entity + "/" + project)

    data_all = {}
    for run in runs:

        data_single = run.history(samples=10000, keys=["test_loss", "message_loss"])
        if "test_loss" not in data_single.columns:
            continue
        if run.config["pruning_step"] != pruning_step:
            continue
        dataset = run.config["dataset"]["name"]
        partial_layer_dropout_prob = run.config["model"]["partial_layer_dropout_prob"]
        message_loss = data_single["message_loss"]
        if dataset not in data_all:
            data_all[dataset] = {}
        if partial_layer_dropout_prob not in data_all[dataset]:
            data_all[dataset][partial_layer_dropout_prob] = {"message_loss": message_loss, "test_loss": []}
        
        data_all[dataset][partial_layer_dropout_prob]["test_loss"].append(data_single["test_loss"])

    def plot_shaded_area(data, label, filename=None):
        message_loss_probs = data["message_loss"]
        loss = np.array(data["test_loss"])
        means = np.mean(loss, axis=0)
        mins = np.min(loss, axis=0)
        maxs = np.max(loss, axis=0)

        # if filename is not None:
        #     df = pd.DataFrame({
        #         "message_loss_prob": [p * 100 for p in message_loss_probs],
        #         "mean_accuracy": means,
        #         "min_accuracy": mins,
        #         "max_accuracy": maxs
        #     })

            # df.to_csv(f"/home/alex/Documents/009_Paper/papers-dsme-nes/federated_learning/images/{filename}.csv", index=False)

        plt.plot(message_loss_probs, means, label=label)
        plt.fill_between(message_loss_probs, np.array(mins), np.array(maxs), alpha=0.3)

    for dataset in ["london_smart_meters_dataset", "traffic_hourly_dataset"]:
        plt.figure(figsize=(10, 6))

        for k in data_all[dataset]:
            plot_shaded_area(data_all[dataset][k], f"Dropout rate: {k}")
        
        
        plt.xlabel("Message loss probability")
        plt.ylabel("Loss")
        plt.title(f"Dataset: {dataset}")
        plt.legend()
        plt.grid(True)

    plt.show()


def plot_pruning():
    api = wandb.Api()
    entity = "distributed_transformer"
    quant = ""
    translate = {"ts_pruning": "Interdevicepruning"}
    probability = 0.0
    data_all = {}
    for project in translate:  #["femnist", "har", "gestures", "CIFAR10"]:
        runs = api.runs(entity + "/" + project)

        data = {}
        for run in runs:

            data_single = run.history(samples=10000, keys=["test_loss"])
            if "test_loss" not in data_single.columns:
                continue
            if abs(run.config["model"]["partial_layer_dropout_prob"] - probability) > 0.01:
                continue
            dataset = run.config["dataset"]["name"]
            if dataset not in data:
                data[dataset] = {}
            num_communication = None
            if translate[project] == "Pruning":
                num_communication = run.config["model"]["num_features_attention"] / 8
            elif translate[project] == "Interdevicepruning":
                if "pruning_step" not in run.config:
                    continue
                num_communication = 16 * (1 - run.config["pruning_step"] * run.config["per_step_pruning_ratio"])
            if num_communication not in data[dataset]:
                data[dataset][num_communication] = [data_single["test_loss"].iloc[0]]
            else:
                data[dataset][num_communication].append(data_single["test_loss"].iloc[0])
        
        data_all[translate[project]] = data

    def plot_shaded_area(data, label, filename=None):
        message_loss_probs = sorted(data.keys())
        accuracies = [data[prob] for prob in message_loss_probs]
        means = [np.mean(acc) for acc in accuracies]
        mins = [np.min(acc) for acc in accuracies]
        maxs = [np.max(acc) for acc in accuracies]

        if filename is not None:
            df = pd.DataFrame({
                "message_loss_prob": [p * 100 for p in message_loss_probs],
                "mean_accuracy": means,
                "min_accuracy": mins,
                "max_accuracy": maxs
            })

            # df.to_csv(f"/home/alex/Documents/009_Paper/papers-dsme-nes/federated_learning/images/{filename}.csv", index=False)

        plt.plot(message_loss_probs, means, label=label)
        plt.fill_between(message_loss_probs, np.array(mins), np.array(maxs), alpha=0.3)

    for dataset in ["london_smart_meters_dataset", "traffic_hourly_dataset"]:
        plt.figure(figsize=(10, 6))

        for k in data_all:
            print(data_all[k][dataset])
            plot_shaded_area(data_all[k][dataset], k)
        
        
        plt.xlabel("Num communication")
        plt.ylabel("Loss")
        plt.title(f"Dataset: {dataset}")
        plt.legend()
        plt.grid(True)

    plt.show()

if __name__ == '__main__':
    # plot_foundational_model_pruning()
    plot_foundational_model_message_loss()
    # plot_message_loss()
    # plot_pruning()
