from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from snntorch.functional.loss import ce_count_loss, ce_rate_loss
from snntorch import utils
import numpy as np
import logging
import torch as th
from utils import device, is_spiking
from itertools import product
from models import *
import copy

logger = logging.getLogger('MSP_Project')

def print_batch_accuracy(model, data, targets, train=False, spiking=True):
    """
    Calculate and print the accuracy of a model on a single minibatch of data.
    Parameters:
      model (nn.Module): The neural network model.
      data (torch.Tensor): The input data.
      targets (torch.Tensor): The target labels.
      train (bool, optional): Whether the function is called during training. Default is False.
      spiking (bool, optional): Whether the model uses spiking neurons. Default is True.
    """
    
    if spiking:
        output, _ = model(data)
        _, idx = output.sum(dim=0).max(1)
    else:
        output = model(data)
        _, idx = output.max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        logger.debug(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        logger.debug(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def reset_weights(model, verbose=False):
  """
  Reset the weights of a model to avoid weight leakage.
  Parameters:
    model (nn.Module): The model whose weights need to be reset.
    verbose (bool, optional): Whether to print debug information. Defaults to False.
  """
  for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        if verbose:
            logger.debug(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

# this function is based on https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
def k_fold_cross_validation(dataset, model, k=10, num_epochs=5, optimizer=None, batch_size=16, verbose=False):
  """
  Perform k-fold cross validation on a given dataset using a specified model.
  Parameters:
    dataset (torch.utils.data.Dataset): The dataset to perform cross validation on.
    model (torch.nn.Module): The model to evaluate during cross validation.
    k (int, optional): The number of folds for cross validation. Default is 10.
    num_epochs (int, optional): The number of epochs to train the model for each fold. Default is 5.
    optimizer (torch.optim.Optimizer, optional): The optimizer to use for training the model. If None, Adam optimizer will be used. Default is None.
    batch_size (int, optional): The batch size for training and testing. Default is 16.
    verbose (bool, optional): Whether to print additional information during training and testing. Default is False.
  Returns:
    average (float): The average accuracy across all folds.
    loss_record (list): A list of training losses for each fold.
  """

  spiking = is_spiking(model)
  if spiking:
    loss_function = ce_rate_loss()
  else:
    loss_function = th.nn.CrossEntropyLoss()
  kfold = KFold(n_splits=k, shuffle=True)
  results = {} # keep track of accuracies, probably we'll want to track f1 as well

  loss_record = []
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = th.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = th.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = th.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=train_subsampler)
    testloader = th.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=test_subsampler)
    model.apply(reset_weights)

    # Initialize optimizer
    if optimizer is None:
      optimizer = th.optim.Adam(model.parameters())

    # Run the training loop for defined number of epochs
    training_loss_at_fold = []
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      total_loss_for_epoch = 0.0
      current_loss = 0.0
      model.train()
      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):

        # Get inputs
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        loss = th.zeros((1), dtype=th.float, device=device)
        # Perform forward pass
        if spiking:
          #utils.reset(model)
          spk_rec, mem_rec = model(inputs)
          loss += loss_function(spk_rec, targets)
        else:
          outputs = model(inputs)
          loss += loss_function(outputs, targets)

        if verbose:
          print_batch_accuracy(model, inputs, targets, train=True, spiking=spiking)

        optimizer.zero_grad()
        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        total_loss_for_epoch += loss.item()
        if i % 10 == 9:
            if verbose:
                logger.debug('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 10))
            current_loss = 0.0
      training_loss_at_fold.append(total_loss_for_epoch/len(trainloader))
      loss_record.append(training_loss_at_fold)

    # Process is complete.
    #print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Saving the model
    #save_path = f'./model-fold-{fold}.pth'
    #th.save(model.state_dict(), save_path)

    # Evaluation for this fold
    correct, total = 0, 0
    with th.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):

        # Get inputs
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Generate outputs


        # Set total and correct
        if spiking:
          test_spk, _ = model(inputs)
          _, predicted = test_spk.sum(dim=0).max(1)
        else:
          outputs = model(inputs)
          _, predicted = th.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

      # Print accuracy
      if verbose:
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
      results[fold] = 100.0 * (correct / total)

  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  average = sum / len(results.items())
  print(f'Average: {average} %')
  return average, loss_record

def hyperparameter_search(train_set, using_mnist=False, spiking_model=True, num_epochs=5, batch_size=16, k=10):
    """
    Perform hyperparameter search for training and testing a model.
    Parameters:
      train_set (Dataset): The training dataset.
      using_mnist (bool, optional): Whether to use MNIST dataset. Defaults to False.
      spiking_model (bool, optional): Whether to use spiking model. Defaults to True.
      num_epochs (int, optional): Number of training epochs. Defaults to 5.
      batch_size (int, optional): Batch size for training. Defaults to 16.
      k (int, optional): Number of folds for k-fold cross validation. Defaults to 10.
    Returns:
      tuple: A tuple containing the best hyperparameters, best accuracy, loss record of best models, and the best model.
    """
    
    learning_rates = [1e-3, 1e-4, 1e-5]
    weight_decays = [0, 1e-3, 1e-4]

    hyperparameter_types = [learning_rates, weight_decays]
    if spiking_model:
        betas = [0.5, 0.8, 0.9]
        hyperparameter_types.append(betas)
    
    hyperparameter_combinations = list(product(*hyperparameter_types))
    logger.debug(f"Hyperparameter combinations: {hyperparameter_combinations}")

    best_accuracy = 0
    best_hyperparameters = None
    best_models_loss_record = None
    best_model = None
    
    for i, hyperparameter in enumerate(hyperparameter_combinations):
        logger.debug(f"Hyperparameter combination {i+1}/{len(hyperparameter_combinations)}: {hyperparameter}")
        learning_rate = hyperparameter[0]
        weight_decay = hyperparameter[1]
        if spiking_model:
            beta = hyperparameter[2]
            model = SpikingCNN(using_mnist=using_mnist, beta=beta)
        else:
            model = PyTorchCNN(using_mnist=using_mnist)
        model = model.to(device)
        optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        mean_accuracy, loss_record = k_fold_cross_validation(train_set, model, k=k, num_epochs=num_epochs, optimizer=optimizer, batch_size=batch_size, verbose=True)
        logger.debug(f"Mean accuracy: {mean_accuracy}")
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_hyperparameters = hyperparameter
            best_models_loss_record = loss_record
            best_model = copy.deepcopy(model)

    logger.debug(f"Best hyperparameters: {best_hyperparameters}")
    logger.debug(f"Best accuracy: {best_accuracy}")
    return best_hyperparameters, best_accuracy, best_models_loss_record, best_model

def train_and_test(model, optimizer, train_set, test_set, num_epochs=5, batch_size=16, verbose=False):
    """
    Trains and tests a given model using the specified optimizer, training set, and test set.
    Parameters:
      model (torch.nn.Module): The model to be trained and tested.
      optimizer (torch.optim.Optimizer): The optimizer used for training the model.
      train_set (torch.utils.data.Dataset): The training dataset.
      test_set (torch.utils.data.Dataset): The test dataset.
      num_epochs (int, optional): The number of epochs to train the model (default is 5).
      batch_size (int, optional): The batch size used for training and testing (default is 16).
      verbose (bool, optional): Whether to print verbose training and testing information (default is False).
    Returns:
      tuple: A tuple containing the training losses, training accuracies, testing losses, and testing accuracies.
    """
    
    spiking = is_spiking(model)
    if spiking:
        loss_function = ce_rate_loss()
    else:
        loss_function = th.nn.CrossEntropyLoss()
    trainloader = th.utils.data.DataLoader(
                      train_set,
                      batch_size=batch_size)
   
    if optimizer is None:
        optimizer = th.optim.Adam(model.parameters())

    training_losses, training_accuracies, testing_losses, testing_accuracies = [], [], [], []
    for epoch in range(0, num_epochs):
        training_loss = 0.0
        training_accuracy = 0.0
        total, correct = 0, 0
        model.train()
        for _, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = 0.0

            optimizer.zero_grad()
            if spiking:
                spk_rec = model(inputs)[0]
                loss += loss_function(spk_rec, targets)
                _, predicted = spk_rec.sum(dim=0).max(1)
            else:
                outputs = model(inputs)
                loss += loss_function(outputs, targets)
                _, predicted = th.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_accuracy = 100 * correct / total
        training_loss /= len(trainloader)
        training_accuracies.append(training_accuracy)
        training_losses.append(training_loss)
        if verbose:
            logger.info(f"Epoch {epoch+1} training loss: {training_loss:.2f}, accuracy: {training_accuracy:.2f}%")
        testing_loss = 0.0
        testing_accuracy = 0.0
        total, correct = 0, 0
        model.eval()
        with th.no_grad():
            testloader = th.utils.data.DataLoader(test_set, batch_size=batch_size)
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                if spiking:
                    test_spk = model(inputs)[0]
                    testing_loss += loss_function(test_spk, targets)
                    _, predicted = test_spk.sum(dim=0).max(1)
                else:
                    outputs = model(inputs)
                    testing_loss += loss_function(outputs, targets)
                    _, predicted = th.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            testing_accuracy = 100 * correct / total
            testing_loss /= len(testloader)
            testing_loss = testing_loss.item()
            testing_accuracies.append(testing_accuracy)
            testing_losses.append(testing_loss)
            if verbose:
                logger.info(f"Epoch {epoch+1} testing loss: {testing_loss:.2f}, accuracy: {testing_accuracy:.2f}%")

    return training_losses, training_accuracies, testing_losses, testing_accuracies

def get_roc_metrics(model, test_data, batch_size=16):
    """
    Calculates the Receiver Operating Characteristic (ROC) metrics for a given model and test data.
    Parameters:
      model (torch.nn.Module): The model to evaluate.
      test_data (torch.utils.data.Dataset): The test data.
      batch_size (int, optional): The batch size for data loading. Default is 16.
    Returns:
      fpr (numpy.ndarray): The False Positive Rate values.
      tpr (numpy.ndarray): The True Positive Rate values.
      thresholds (numpy.ndarray): The thresholds values.
      auc (float): The Area Under the ROC Curve (AUC) value.
    """
    
    spiking = is_spiking
    model.eval()
    with th.no_grad():
        testloader = th.utils.data.DataLoader(test_data, batch_size=batch_size)
        all_targets = []
        all_probabilities_or_spike_counts = []
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            if spiking:
                test_spk = model(inputs)[0]
                # sum spikes over all time steps for each class
                probabilities = test_spk.sum(dim=0)
            else:
                outputs = model(inputs)
                probabilities = th.nn.functional.softmax(outputs, dim=1)
            all_targets.append(targets)
            # grab just the probabilities for the positive class
            probabilities = probabilities[:, 1]
            all_probabilities_or_spike_counts.append(probabilities)
        all_targets = th.cat(all_targets)
        all_probabilities_or_spike_counts = th.cat(all_probabilities_or_spike_counts)

        fpr, tpr, thresholds = roc_curve(all_targets.cpu().numpy(), all_probabilities_or_spike_counts.cpu().numpy())
        auc = roc_auc_score(all_targets.cpu().numpy(), all_probabilities_or_spike_counts.cpu().numpy())
        return fpr, tpr, thresholds, auc
    
def run_spiking_inference(model, data_point):
    """
    Run spiking inference on a given model and data point.
    Parameters:
      model (torch.nn.Module): The spiking model to be evaluated.
      data_point (torch.Tensor): The input data point for inference.
    Returns:
      Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the predicted class label,
      spike recordings for the fourth layer (spk_rec4), membrane potential recordings for the fourth layer (mem_rec_4),
      spike recordings for the third layer (spk_rec_3), and membrane potential recordings for the third layer (mem_rec_3).
    """
    
    model.eval()
    with th.no_grad():
        data_point = data_point.to(device)
        spk_rec4, mem_rec_4, spk_rec_3, mem_rec_3 = model(data_point)
        _, predicted = spk_rec4.sum(dim=0).max(1)
        return predicted.item(), spk_rec4, mem_rec_4, spk_rec_3, mem_rec_3