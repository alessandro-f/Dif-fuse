import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

sns.set()

import pandas as pd
from utils.storage import load_metrics_dict_from_pt, save_metrics_dict_in_pt


def compute_accuracy(logits, targets):
    if len(logits) > 1:
        acc = (targets == logits.argmax(-1)).float().detach().cpu().numpy()
        mean_acc = np.mean(acc)
    elif len(logits) == 1: #proiding prediction directly
        acc = (targets == logits).float().detach().cpu().numpy()
        mean_acc = np.mean(acc)
    else:
        mean_acc = np.nan
    return mean_acc

def compute_confusion_matrix_raw(logits, targets, classes=2):
    """

    Args:
        logits : raw output from NN
        targets : true labels
        classes: classes to predict
    Returns:

    """
    pred = logits.argmax(-1)
    stacked = torch.stack((targets, pred), dim=1)
    cmt = torch.zeros(2, 2, dtype=torch.int8)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    return cmt


def compute_sensitivity(epoch_pred_confusion):
    """
    sensitivity = TP/(TP + FN)
    Args:
        epoch_pred_confusion: string array of the confusion matrix ["TP", "TN", ...]
    """
    #
    TP_num = epoch_pred_confusion.count("TP")
    FN_num = epoch_pred_confusion.count("FN")
    return float(TP_num / (TP_num + FN_num)) if (TP_num + FN_num) > 0 else 0.0


def compute_specificity(epoch_pred_confusion):
    """
    specificity = TN/(TN + FP)
    Args:
        epoch_pred_confusion: epoch_pred_confusion: string array of the confusion matrix ["TP", "TN", ...]
    """
    TN_num = epoch_pred_confusion.count("TN")
    FP_num = epoch_pred_confusion.count("FP")
    return float(TN_num / (TN_num + FP_num)) if (TN_num + FP_num) > 0 else 0.0


def compute_acc_from_confusion_list(confusion_list):
    """
    compute accuracy, sensitivity, specificity and summary of confusion matrix
    Args:
        confusion_list: data frame with format ["TP", "FN", "TN", ...]
    """
    TP = confusion_list.count("TP")
    TN = confusion_list.count("TN")
    FP = confusion_list.count("FP")
    FN = confusion_list.count("FN")
    return (
        float((TP + TN) / (TP + TN + FP + FN))
        if (TP + TN + FP + FN) > 0
        else 0.0
    )

def compute_confusion_matrix_MTL(ids, pred_dict, targets_dict):
    pred_MTL_confusion_dict = {}
    for task_name, pred_task in pred_dict.items():
        pred_MTL_confusion_dict[task_name] = \
            compute_confusion_matrix(ids, pred_task, targets_dict[task_name])
    return pred_MTL_confusion_dict
def compute_confusion_matrix(ids, preds, targets):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    pred_dict = {}
    for id, each_pred, true_label in zip(ids, preds, targets):
        pred_dict[id] = {}
        if each_pred == true_label and true_label == 1:
            TP += 1
            confusion_result = "TP"
        elif each_pred == true_label and true_label == 0:
            TN += 1
            confusion_result = "TN"
        elif each_pred == 1 and true_label == 0:
            FP += 1
            confusion_result = "FP"
        elif each_pred == 0 and true_label == 1:
            FN += 1
            confusion_result = "FN"
        pred_dict[id]["pred"] = each_pred
        pred_dict[id]["pred_confusion"] = confusion_result
    # output_string = "".join(" TP: ",str(TP), " TN: ",str(TN), " FP: ", str(FP), " FN: ", str(FN))
    return pred_dict


class MetricTracker:
    def __init__(
        self,
        tracker_name,
        metrics_to_track=None,
        load=True,
        path="",
        log_dict = {},
        confusion_path="",
        sensitivity_path="",
        specificity_path="",
        task_dict=None
    ):
        if metrics_to_track is None:
            metrics_to_track = {
                "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
                "accuracy": compute_accuracy,
            }
        self.metrics_to_track = metrics_to_track
        self.tracker_name = tracker_name
        self.metrics = {"epochs": [], "iterations": []}
        self.path = path
        self.log_dict = log_dict
        self.confusion_path = confusion_path
        self.sensitivity_path = sensitivity_path
        self.specificity_path = specificity_path
        self.task_dict = task_dict
        self.load = load

        if self.load and os.path.isfile(path):
            metrics_from_file = load_metrics_dict_from_pt(path=path)
            self.metrics = metrics_from_file

    def push(self, epoch, iteration, logits, targets):
        self.metrics["epochs"].append(epoch)
        self.metrics["iterations"].append(iteration)

        if isinstance(logits, dict):
            for task_name, logit_item, target_item in zip(logits.keys(),
                                                          logits.values(),
                                                          targets.values()):
                for k, fnc in self.metrics_to_track.items():
                    if not f"{task_name}_{k}" in self.metrics.keys():
                        self.metrics[f"{task_name}_{k}"] = []

                    self.metrics[f"{task_name}_{k}"].append(fnc(logit_item,
                                                                    target_item))
        else:
            for k, fnc in self.metrics_to_track.items():
                if not f"{k}" in self.metrics.keys():
                    self.metrics[f"{k}"] = []
                self.metrics[k].append(fnc(logits, targets))

    def save(self):
        save_metrics_dict_in_pt(
            path=self.path, metrics_dict=self.metrics, overwrite=True
        )

    def collect_per_epoch(self):
        epoch_metrics = {"epochs": []}
        for k, _ in self.metrics.items():
            if k not in ["epochs", 'iterations']:
                epoch_metrics["{}_mean".format(k)] = []
                epoch_metrics["{}_std".format(k)] = []

        epochs = self.metrics["epochs"]
        unique_epochs = np.unique(epochs)
        epoch_metrics["epochs"] = unique_epochs

        for k, v in self.metrics.items():
            v = np.array(v)
            if k not in ["iterations", "epochs"] and len(v) > 0:
                for this_epoch in unique_epochs:
                    where_metrics = epochs == this_epoch
                    v_mean = np.mean(v[where_metrics])
                    v_std = np.std(v[where_metrics])
                    epoch_metrics["{}_mean".format(k)].append(v_mean)
                    epoch_metrics["{}_std".format(k)].append(v_std)
            elif k not in ["iterations", "epochs"] and len(v) == 0:
                epoch_metrics["{}_mean".format(k)].append(np.nan)
                epoch_metrics["{}_std".format(k)].append(np.nan)
        if os.path.isfile(self.sensitivity_path):
            sensitivity_df = pd.read_csv(self.sensitivity_path)
            epoch_metrics["sensitivity"] = sensitivity_df["sensitivity"].tolist()

        if os.path.isfile(self.specificity_path):
            specificity_df = pd.read_csv(self.specificity_path)
            epoch_metrics["specificity"] = specificity_df["specificity"].tolist()
        return epoch_metrics

    def get_best_epoch_for_metric(self, metric_name, evaluation_metric=np.argmax):

        best_metric = evaluation_metric(self.collect_per_epoch()[metric_name])
        print(f"metric_name: {metric_name} best: {best_metric}")

        return best_metric

    def plot(self, path, plot_std_dev=True):
        epoch_metrics = self.collect_per_epoch()
        # print(epoch_metrics)

        x = np.array(epoch_metrics["epochs"])
        keys = [
            k
            for k, _ in epoch_metrics.items()
            if k not in ["epochs", "sensitivity", "specificity"]
        ]
        reduced_keys = []

        for key in keys:
            reduced_key = key.replace("_mean", "").replace("_std", "")
            if reduced_key not in reduced_keys:
                reduced_keys.append(reduced_key)
        num_axes = len(reduced_keys)
        nrow = 2
        ncol = int(np.ceil(num_axes / nrow))
        fig = plt.figure(figsize=(5 * nrow, 5 * ncol))
        max_acc = {}
        max_acc_epoch = {}
        for k, v in epoch_metrics.items():
            if "mean" in k:
                v = np.array(v)
                max_acc[k] = np.max(v)
                max_acc_epoch[k] = np.argmax(v)

        if len(list(max_acc_epoch.keys())) == 1:#multi task
            epoch_sensitivity = (
                epoch_metrics["sensitivity"][max_acc_epoch]
                if "sensitivity" in epoch_metrics.keys()
                else 0.0
            )
            epoch_specificity = (
                epoch_metrics["specificity"][max_acc_epoch]
                if "specificity" in epoch_metrics.keys()
                else 0.0
            )

        for pi, key in enumerate(reduced_keys):
            ax = fig.add_subplot(ncol, nrow, pi + 1)
            y_mean = np.array(epoch_metrics[key + "_mean"])
            y_std = np.array(epoch_metrics[key + "_std"])
            if plot_std_dev:
                ax.fill_between(
                    x,
                    y_mean - y_std,
                    y_mean + y_std,
                    np.ones_like(x) == 1,
                    color="g" if "entropy" in key else "m",
                    alpha=0.1,
                )
            ax.plot(x, y_mean, "g-" if "entropy" in key else "m-", alpha=0.9)
            ax.set_ylabel(key)
            ax.set_xlabel("epochs")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        del fig


def plot_single_metrics(x, y, ylabel, xlabel="epochs", title=None, save_path=None,show=False):
    fig, ax = plt.subplots()
    ax.plot(x, y, "b-", alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    if show:
        plt.show()
    plt.close(fig)
    del fig

def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


