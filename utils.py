import torch.nn as nn
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_metrics(log_dir: str):
    """
    plots metrics from log directory that includes csv log file
    args:
        log_dir: path/to/log_dir
    returns:
        None
    """
    dirs = os.listdir(log_dir)
    logfiles = [os.path.join(log_dir, d) for d in dirs if d.endswith(".csv")]
    assert (
        len(logfiles) == 1
    ), "log_dir should contain only one csv log file, please check"
    metrics = pd.read_csv(logfiles[0])

    org_columns = metrics.columns
    org_columns = [c for c in org_columns if "test" not in c]
    epoch_columns = [c for c in org_columns if "epoch" in c]
    step_columns = [c for c in org_columns if "step" in c]
    other_columns = list(set(org_columns) - set(epoch_columns) - set(step_columns)) + [
        "step"
    ]
    epoch_metrics = metrics[epoch_columns].set_index("epoch")
    step_metrics = metrics[step_columns].set_index("step")

    other_metrics = None
    if other_columns:
        other_metrics = metrics[other_columns].set_index("step")
        other_metrics_names = other_metrics.columns

    save_dir = os.path.join(log_dir, "metric_curves")
    os.makedirs(save_dir, exist_ok=True)
    epoch_metrics_names = epoch_metrics.columns
    step_metrics_names = step_metrics.columns
    for name in epoch_metrics_names:
        fig, ax = plt.subplots(1, 1)
        ax = epoch_metrics[name].dropna().plot(title=name, legend=True)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.close()
    for name in step_metrics_names:
        fig, ax = plt.subplots(1, 1)
        ax = step_metrics[name].dropna().plot(title=name, legend=True)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.close()
    if other_metrics is not None:
        for name in other_metrics_names:
            fig, ax = plt.subplots(1, 1)
            ax = other_metrics[name].dropna().plot(title=name, legend=True)
            fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.close()


def save_image(images: list[torch.Tensor], path: str, titles: list[str] = None) -> None:
    """
    args:
        images: a list of 4D-tensors
        path: path to save images
        titles: title of tensors
    returns: None
    """
    images = [make_grid(img).permute(1, 2, 0) for img in images]
    n_samples = len(images)
    if titles:
        assert len(images) == len(titles)
    else:
        titles = [""] * n_samples
    fig, axes = plt.subplots(n_samples, 1)
    for ax, img, tit in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(tit)
    fig.savefig(path, bbox_inches="tight")
    plt.close()
