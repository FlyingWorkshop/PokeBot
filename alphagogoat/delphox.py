from torch import nn
import torch

NUM_MOVES = 4

class Delphox(nn.Module):
    def __init__(self):
        super().__init__()

        n = 10  # TODO: fix
        self.stack = nn.Sequential(
            nn.Softmax(n, )
        )


    def forward(self, x):
        return