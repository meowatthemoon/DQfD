import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 40)
        self.fc2 = nn.Linear(40, 3)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Policy:
    def __init__(self):
        self.predictNet = Network()
        self.targetNet = Network()
        *_, last = self.predictNet.children()
        self.A = list(range(last.out_features))
        self.updateTargetNet()

    def calcQ(self, net, s, A):
        # 1.single state, one or multiple actions
        # 2.muliplte states, one action per state, s must be a list of tensors
        if isinstance(s, torch.Tensor) and s.dim() == 1:  # situation 1
            return torch.Tensor([net(s)[a] for a in A]).squeeze()

        if not isinstance(s, torch.Tensor):  # situation 2
            s = torch.stack(s)
            Q = net(s)
            A = [a[0] for a in A]
            return Q[[i for i in range(len(A))], A]

    def sortedA(self, state):
        net = self.predictNet
        net.eval()
        Q = self.calcQ(net, state, self.A)
        A = [[a] for q, a in sorted(zip(Q, self.A), reverse=True)]
        net.train()
        return A

    def updateTargetNet(self):
        self.targetNet.load_state_dict(self.predictNet.state_dict())
        self.targetNet.eval()
