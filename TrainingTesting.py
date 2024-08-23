from sklearn.model_selection import KFold
from snntorch.functional.loss import ce_count_loss, ce_rate_loss
from snntorch import utils
import numpy as np
import logging
import torch as th
from utils import *

logger = logging.getLogger('MSP_Project')
device = get_device()

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
  spiking = False
  if 'Spiking' in model.__class__.__name__:
    spiking = True
  if spiking:
    loss_function = ce_rate_loss()
  else:
    loss_function = th.nn.CrossEntropyLoss()
  kfold = KFold(n_splits=k, shuffle=True)
  results = {} # keep track of accuracies, probably we'll want to track f1 as well

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
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
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
          loss = loss_function(outputs, targets)

        if verbose:
          print_batch_accuracy(model, inputs, targets, train=True, spiking=spiking)

        optimizer.zero_grad()
        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        if i % 10 == 9:
            if verbose:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 10))
            current_loss = 0.0

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
  return average