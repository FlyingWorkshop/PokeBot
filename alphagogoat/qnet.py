import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from constants import *
from embedder import Embedder

EMB = Embedder()

class QNet(nn.Module):
    def __init__(self, input_size, hidden_layers=2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, input_size//2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_size//2, input_size//4)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(input_size//4, len(MoveEnum) + 1)
    
    def forward(self, curr_state: torch.tensor, action: str):
        """
        Takes in a tensor representing a battle state
        """
        

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)

        return self.out(x)
        
    
