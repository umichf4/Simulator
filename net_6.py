import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimulatorNet(nn.Module):
    def __init__(self, in_num=2, out_num=28):
        super(SimulatorNet, self).__init__()
        self.FCIN = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(in_num, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU()
        )
        self.FC = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU()
        )
        self.FCOUT = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(1000, out_num),
            nn.BatchNorm1d(out_num),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.FCIN(x)
        x = x + self.FC(x)
        x = self.FC(x)
        x = x + self.FC(x)
        x = self.FC(x)
        x = x + self.FC(x)
        x = self.FC(x)
        x = x + self.FC(x)
        x = self.FCOUT(x)
        return x


if __name__ == '__main__':
    import torchsummary

    if torch.cuda.is_available():
        simulator = SimulatorNet().cuda()
    else:
        simulator = SimulatorNet()
    params = list(simulator.parameters())
    print(len(params))
    torchsummary.summary(simulator, tuple([2]))
