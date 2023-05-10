from torch import nn
import torch

NUM_MOVES = 4
NUM_INPUT = 81

class Delphox(nn.Module):
    def __init__(self):
        super().__init__()

        n = 10  # TODO: fix
        self.stack = nn.Sequential(
            nn.Softmax(n, )
        )


    def forward(self, x):

        action = torch.max(self.stack(x))

        return