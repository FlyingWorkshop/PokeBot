import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
import embedder
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

        self.softmax = nn.Softmax(dim=1)

    def forward(self, team1: dict[str: Pokemon], team2: dict[str: Pokemon]):
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


        
        
        

#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x
