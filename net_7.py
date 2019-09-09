import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimulatorNet(nn.Module):
    def __init__(self, in_num=2, out_num=28):
        super(SimulatorNet, self).__init__()
        self.FC = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(in_num, 5000),
            nn.BatchNorm1d(5000),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(5000, 4000),
            nn.BatchNorm1d(4000),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(4000, 3000),
            nn.BatchNorm1d(3000),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(3000, out_num)
        )

    def forward(self, x):
        x = self.FC(x)
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
