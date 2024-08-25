import matplotlib.pyplot as plt
from sklearn.Metrics import RocCurveDisplay

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.show()

def plot_training_testing_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
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
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_scores,

                                  estimator_name='example estimator')

    display.plot()
    plt.show()