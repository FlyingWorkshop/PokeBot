import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
import embedder
from torch.nn.init import xavier_uniform_
import pokedex
#
# from poke_env.environment.battle import Battle
#
TEAM_SIZE = 6


class Delphox(nn.Module):
    def __init__(self, input_size, output_size):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        self.emb = embedder.Embedder()

        self.rnn = nn.LSTM(input_size, output_size, 2)

        self.h0 = torch.randn(2, output_size)
        self.c0 = torch.randn(2, output_size)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, team1: dict[str: Pokemon], team2: dict[str: Pokemon], hidden: tuple[torch.tensor]):
        pokemon = []
        moves = []

        for t1_pokemon in team1.values():
            pokemon.append(self.emb._embed_pokemon(t1_pokemon))
            moves.append(self.emb._embed_moves_from_pokemon(t1_pokemon))
        
        for t2_pokemon in team2.values():
            pokemon.append(self.emb._embed_pokemon(t2_pokemon))
            moves.append(self.emb._embed_moves_from_pokemon(t2_pokemon))

        num_unknown_pokemon = 2 * TEAM_SIZE - len(team1) - len(team2)
        # TODO: edit probs
        pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon), mode='constant', value=-1)
        moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))

        x = torch.cat(pokemon.flatten(), moves.flatten())
        
        x, forward = self.rnn(x, hidden)
        
        x = self.softmax(x)

        return x, forward


def train(data: dict[Battle: tuple]):
    delphox = Delphox(7800, 296 + 1)

    optimizer = torch.optim.Adam(delphox.parameters(), lr=0.001)

    for battle, (team1, team2) in data.items():
        hidden = (torch.randn(2, 296 + 1) , torch.randn(2, 296 + 1))
        output, hidden = delphox(team1, team2, hidden)
        loss = F.cross_entropy(output, battle.outcome) #TODO fix this loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()