import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimulatorNet(nn.Module):
    def __init__(self):
        super(SimulatorNet, self).__init__()
        self.FCIN = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(2, 64 * 64),
            nn.BatchNorm1d(64 * 64),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(64 * 64, 2 * 64 * 64),
            nn.BatchNorm1d(2 * 64 * 64),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(2, 2 * 64 * 64),
            nn.BatchNorm1d(2 * 64 * 64),
            nn.LeakyReLU(0.2)
        )

        self.CONV2D = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(2, 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(4, 4 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(4 * 2, 4 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(4 * 4, 4 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        )

        self.FCOUT = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(4 * 8 * 4 * 4, 4 * 8 * 4 * 4),
            nn.BatchNorm1d(4 * 8 * 4 * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(4 * 8 * 4 * 4, 4 * 8 * 4 * 4),
            nn.BatchNorm1d(4 * 8 * 4 * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(4 * 8 * 4 * 4, 28),
            nn.BatchNorm1d(28),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.FCIN(x)
        x = x.view(-1, 2, 64, 64)
        x = self.CONV2D(x)
        x = x.view(-1, 4 * 8 * 4 * 4)
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
