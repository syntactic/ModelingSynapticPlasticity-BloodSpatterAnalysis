import torch.nn as nn
import torch.nn.functional as F
import torch as th
import snntorch as snn

class PyTorchCNN(nn.Module):
    """
    PyTorchCNN is a convolutional neural network model implemented using PyTorch.
    Parameters:
      using_mnist (bool): Flag indicating whether the model is being used for MNIST dataset or not.
    Attributes:
      num_classes (int): Number of classes in the output layer.
      fully_connected_input_size (int): Size of the input to the fully connected layer.
      conv1 (nn.Conv2d): First convolutional layer.
      pool (nn.MaxPool2d): Max pooling layer.
      conv2 (nn.Conv2d): Second convolutional layer.
      fc1 (nn.Linear): First fully connected layer.
      fc2 (nn.Linear): Second fully connected layer.
    """
    def __init__(self, using_mnist=False):
        self.using_mnist = using_mnist
        if using_mnist:
          self.num_classes = 10
          self.fully_connected_input_size = 16 * 4 * 4
          conv_layer_kernel_size = 5
        else:
          self.num_classes = 2
          conv_layer_kernel_size = 5
          self.fully_connected_input_size = 16 * 13 * 13
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, conv_layer_kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, conv_layer_kernel_size)
        self.fc1 = nn.Linear(self.fully_connected_input_size, 120)
        self.fc2 = nn.Linear(120, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.using_mnist:
          x = x.view(batch_size, -1)
        else:
          x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SpikingCNN(PyTorchCNN):
    """
    A class representing a spiking CNN model.
    Parameters:
      beta (float, optional): The membrane potential decay rate. Defaults to 0.8.
      num_steps (int, optional): The number of time steps for the spiking dynamics. Defaults to 10.
      **kwargs: Additional keyword arguments to be passed to the base class constructor.
    Attributes:
      lif1 (snn.Leaky): The first leaky integrate-and-fire (LIF) neuron layer.
      lif2 (snn.Leaky): The second leaky integrate-and-fire (LIF) neuron layer.
      lif3 (snn.LeakyParallel): The third leaky integrate-and-fire (LIF) neuron layer.
      lif4 (snn.LeakyParallel): The fourth leaky integrate-and-fire (LIF) neuron layer.
    """
    def __init__(self, beta=0.8, num_steps=10, **kwargs):
        self.num_steps = num_steps
        super().__init__(**kwargs)

        # Initialize layers
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.LeakyParallel(input_size=self.fully_connected_input_size, hidden_size=120, beta=beta)
        self.lif4 = snn.LeakyParallel(input_size=120, hidden_size=self.num_classes, beta=beta)

    def forward(self, x):
        batch_size = x.shape[0]
        spk4_rec = []
        mem4_rec = []

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for _ in range(self.num_steps):
            cur1 = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            spk3 = self.lif3(spk2.view(batch_size, -1))
            spk4 = self.lif4(spk3)
            spk4_rec.append(spk4)
            mem4_rec.append(mem2)

        return th.stack(spk4_rec, dim=0), th.stack(mem4_rec, dim=0)
    
class SpikingCNNSerial(SpikingCNN):
    """
    A class representing a non-parallel spiking CNN model.
    Args:
      **kwargs: Additional keyword arguments for the parent class.
        beta (float): The membrane potential decay rate.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lif3 = snn.Leaky(beta=kwargs['beta'])
        self.lif4 = snn.Leaky(beta=kwargs['beta'])
        self.fc2 = nn.Linear(120, self.num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        spk4_rec = []
        spk3_rec = []
        mem3_rec = []
        mem4_rec = []

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        for _ in range(self.num_steps):
            cur1 = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        # return spikes and membrane potentials of last two leaky integrators
        return th.stack(spk4_rec, dim=0), th.stack(mem4_rec, dim=0), th.stack(spk3_rec, dim=0), th.stack(mem3_rec, dim=0)
   