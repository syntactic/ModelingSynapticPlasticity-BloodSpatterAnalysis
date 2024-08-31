# ModelingSynapticPlasticity-BloodSpatterAnalysis
Project for Modeling Synaptic Plasticity course at University of Osnabrück. Blood spatter cause identification using spiking neural networks.

Group 15: Timothy Ho, Nina Ma

## Dependencies
The primary libraries needed to run the code in this repository are: `torch`, `torchvision`, `numpy`, `pandas`, `snntorch`, and `matplotlib`. 

There is an included `requirements.txt` file and it can be used to install all necessary libraries:

```
pip install -r requirements.txt
```

## Usage
The best way to view the work in this project is to run the included Jupyter [notebook](https://github.com/syntactic/ModelingSynapticPlasticity-BloodSpatterAnalysis/blob/main/project.ipynb).

There is also a `main.py` file that can be run without any arguments, which will run a hyperparameter search over a small space to determine the optimal parameters within that space, and then train a model using the full training set and those parameters, and report the losses and accuracies.

```
python main.py
```

Note that the flag `SPIKING_MODEL` is set to `False` in `main.py`, because running hyperparameter search over the non-spiking model is much faster. If you would like to do hyperparameter search for a spiking model, set that flag to `True`.

There is also a `USING_MNIST` flag which is set to `False`. MNIST is an "easy" dataset used to debug issues that the team encountered when training the models. It can still be switched to `True` for both the main script and the notebook if desired. However, plotting of neural activity in the Jupyter notebook does not work because the classes used in that function are hard-coded as "Blunt Trauma" and "Firearm". 

## Files Explanation
`models.py` - contains three neural network architectures: 
* `PyTorchCNN`: non-spiking convolutional neural network with ReLU activations
* `SpikingCNNSerial`: spiking convolutional neural network with `Leaky` layers
* `SpikingCNN`: spiking convolutional neural network with `LeakyParallel` layers. Used during hyperparameter search to speed up processing.

`data.py` - contains functions related to downloading the datasets, creating Dataset objects out of them, and measuring class distributions.

`transformations.py` - contains image transformations for data augmentation and preprocessing.

`exploratory_data_analysis.py` - contains functions related to plotting the image data or class data.

`plotting.py` - contains functions related to plotting model evaluation and inference.

`BloodDataset.py` - definition of the custom Dataset class used in this project. The primary customizations are:
* transformation to apply at construction (downsampling)
* data augmentation to apply on method call

`training_testing.py` - contains code for k-fold cross validation, training, testing, and inference.

`utils.py` - contains functions that don't fit in the other files.

`project.ipynb` - Jupyter notebook containing a demonstration of the main functionalities of this project.

`main.py` - Main script that runs the dataset download, division, training, and testing. This script is dispreferred to the notebook.

`training_nonspiking_cnn.txt` - Log of hyperparameter search for non-spiking CNN using 10-fold cross validation.

`training_spiking_cnn.txt` - Log of hyperparameter search for spiking CNN using 10-fold cross validation.