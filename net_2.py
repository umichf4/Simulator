import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimulatorNet(nn.Module):
    def __init__(self, in_num=2, out_num=17):
        super(SimulatorNet, self).__init__()
        self.FC = nn.Sequential(
             # ------------------------------------------------------
            nn.Linear(in_num, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Linear(5000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Linear(4000, 3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Linear(3000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Linear(1000, out_num)
        )

    def forward(self, x):
        x = self.FC(x)
        return x


# class FirstBlinear(nn.Module):
#     """Applies a linear transformation to the incoming data: :math:`y = e^TWe + e^TM(e*e) + V[e (e*e)]^T + b`
#     """
#     __constants__ = ['in_features', 'out_features']

#     def __init__(self, in_features, out_features):
#         super(FirstBlinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.W = nn.Parameter(torch.Tensor(out_features, in_features, in_features), requires_grad=True)
#         self.M = nn.Parameter(torch.Tensor(out_features, in_features, in_features), requires_grad=True)
#         self.V = nn.Parameter(torch.Tensor(2 * in_features, out_features), requires_grad=True)
#         self.b = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
#         self.reset_parameters()

#     def reset_parameters(self):
#         bound = 1 / math.sqrt(self.W.size(1))
#         nn.init.uniform_(self.W, -bound, bound)
#         nn.init.uniform_(self.M, -bound, bound)
#         nn.init.uniform_(self.V, -bound, bound)
#         nn.init.uniform_(self.b, -bound, bound)

#     def forward(self, input):
#         input_square = torch.mul(input, input)
#         return F.bilinear(input, input, self.W) + F.bilinear(input, input_square, self.M) + \
#             torch.mm(torch.cat((input, input_square), 1), self.V) + self.b

#     def extra_repr(self):
#         return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


if __name__ == '__main__':
    import torchsummary

    if torch.cuda.is_available():
        simulator = SimulatorNet().cuda()
    else:
        simulator = SimulatorNet()
    params = list(simulator.parameters())
    print(len(params))
    torchsummary.summary(simulator, tuple([2]))