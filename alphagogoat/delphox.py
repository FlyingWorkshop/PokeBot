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

        #broken up network here:
        #-------------------------
        # pass matrix through a linear layer (flaten maybe), to get a distributioin over all possible actions.
        # then set all invalid options to -inf
        # then softmax, select action


    def forward(self, x):
        return