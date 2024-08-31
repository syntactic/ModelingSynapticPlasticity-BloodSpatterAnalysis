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