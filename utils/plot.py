import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", cmap="Blues", path=None):
    """
    Plots the confusion matrix with the provided labels.

    Args:
        cm (np.ndarray): The confusion matrix (2D array).
        labels (list): List of label names to display on the axes.
        title (str): Title of the plot.
        cmap (str or matplotlib.colors.Colormap): Colormap for the heatmap.
    
    Returns:
        None: Displays the confusion matrix plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

def plot_metrics_epoch(metrics, title="Training Metrics", xlabel="Epoch", ylabel="Value", figsize=(10, 6),path=None):
    """
    Plots trend graphs for all numerical lists in a dictionary over epochs.

    Args:
        metrics (dict): Dictionary where keys are metric names (e.g., "train_loss", "val_loss")
                        and values are lists representing values per epoch.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis (default: "Epoch").
        ylabel (str): Label for the y-axis (default: "Value").
        figsize (tuple): Size of the figure (default: (10, 6)).
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(next(iter(metrics.values()))) + 1)  # Infer number of epochs from first key
    
    for key, values in metrics.items():
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):  # Ensure it's numerical data
            sns.lineplot(x=epochs, y=values, marker="o", label=key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
