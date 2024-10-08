import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def plot_loss(losses):
    """
    Plots the loss values over epochs.
    Parameters:
        losses (list): A list of loss values.
    """
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.show()

def plot_training_testing_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    """
    Plots the training and testing metrics (loss and accuracy) over epochs.
    Parameters:
        train_losses (list): List of training losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
        test_losses (list): List of testing losses for each epoch.
        test_accuracies (list): List of testing accuracies for each epoch.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(test_losses, label='Testing Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.legend()

    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(test_accuracies, label='Testing Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, auc_scores):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    Parameters:
        fpr (array-like): False Positive Rate values.
        tpr (array-like): True Positive Rate values.
        auc_scores (float): Area Under the ROC Curve (AUC) score.
    """

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_scores,

                                  estimator_name='example estimator')

    display.plot()
    plt.show()

# the following function was taken from the course's LIF notebook
def plot_network_activity(voltage, spikes, name=None):
    """
    Plots the network activity of neurons.
    Parameters:
        voltage (numpy.ndarray): 2D array representing the voltage activity of neurons.
        spikes (numpy.ndarray): 2D array representing the firing activity of neurons.
        name (str, optional): Name of the plot (default: None).
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    if name is not None:
        fig.suptitle(name, fontsize=16)
    h, w = voltage.shape

    im_voltage = ax[0].imshow(voltage)
    ax[0].set_title("Voltage", fontsize=14)
    ax[0].set_ylabel("Neurons", fontsize=14)
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_aspect(w / h)
    fig.colorbar(im_voltage, ax=ax[0])

    im_spikes = ax[1].imshow(spikes)
    ax[1].imshow(spikes)
    ax[1].set_title("Firing activity", fontsize=14)
    ax[1].set_ylabel("Neurons", fontsize=14)
    ax[1].set_xlabel("Time [ms]")
    ax[1].set_aspect(w / h)

    fig.tight_layout()
    plt.show()
