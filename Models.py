import torch.nn as nn
import torch.nn.functional as F
import torch as th
import snntorch as snn

class PyTorchCNN(nn.Module):
    def __init__(self, using_mnist=False):
        self.using_mnist = using_mnist
        if using_mnist:
          num_classes = 10
        else:
          num_classes = 2
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if self.using_mnist:
          self.fc1 = nn.Linear(16 * 4 * 4, 120)
        else:
          self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.using_mnist:
          x = x.view(-1, 16 * 4 * 4)
        else:
          x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SpikingCNN(nn.Module):
    def __init__(self, beta=0.8, num_steps=50):
        self.num_steps = num_steps
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta)
        #self.fc1 = nn.Linear(64*29*29, 2)
        #self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        #self.lif3 = snn.LeakyParallel(input_size=64*29*29, hidden_size=2, beta=beta)
        self.lif3 = snn.LeakyParallel(input_size=64*4*4, hidden_size=2, beta=beta)

    def forward(self, x):
        batch_size = x.shape[0]
        spk3_rec = []
        mem3_rec = []

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        #mem3 = self.lif3.init_leaky()

        for step in range(self.num_steps):
            cur1 = F.max_pool2d(self.conv1(x), 2)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)

            #cur3 = self.fc1(spk2.view(batch_size, -1))
            #spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.lif3(spk2.view(batch_size, -1))
            spk3_rec.append(spk3)
            mem3_rec.append(mem2)

        return th.stack(spk3_rec, dim=0), th.stack(mem3_rec, dim=0)