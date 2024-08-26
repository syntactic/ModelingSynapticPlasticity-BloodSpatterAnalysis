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
    if spiking:
        output, _ = model(data)
        _, idx = output.sum(dim=0).max(1)
    else:
        output = model(data)
        _, idx = output.max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def reset_weights(model, verbose=False):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        if verbose:
            print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

# this function is based on https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
def k_fold_cross_validation(dataset, model, k=10, num_epochs=5, optimizer=None, batch_size=16, verbose=False):
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
                print('Loss after mini-batch %5d: %.3f' %
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
            print(f"Epoch {epoch+1} loss: {training_loss:.2f}, accuracy: {training_accuracy:.2f}%")
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
                print(f"Epoch {epoch+1} test loss: {testing_loss:.2f}, accuracy: {testing_accuracy:.2f}%")

    return training_losses, training_accuracies, testing_losses, testing_accuracies

def get_roc_metrics(model, test_data, batch_size=16):
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
    model.eval()
    with th.no_grad():
        data_point = data_point.to(device)
        spk_rec4, mem_rec_4, spk_rec_3, mem_rec_3 = model(data_point)
        _, predicted = spk_rec4.sum(dim=0).max(1)
        return predicted.item(), spk_rec4, mem_rec_4, spk_rec_3, mem_rec_3