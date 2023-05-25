import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
import embedder
from torch.nn.init import xavier_uniform_
from pokedex import *
from catalogs import *
#import data_labeler
from small_dataset import *
from constants import *


class Delphox(nn.Module):
    def __init__(self, input_size, output_size):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        self.emb = embedder.Embedder()

        self.rnn = nn.LSTM(input_size, output_size, NUM_HIDDEN_LAYERS)

        self.softmax = nn.Softmax(dim = 0)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, team1: dict[str: Pokemon], team2: dict[str: Pokemon], hidden: tuple[torch.tensor]):
        pokemon = []
        moves = []

        for t1_pokemon in team1.values():
            pokemon.append(self.emb._embed_pokemon(t1_pokemon))
            moves.append(self.emb._embed_moves_from_pokemon(t1_pokemon))
        
        for t2_pokemon in team2.values():
            pokemon.append(self.emb._embed_pokemon(t2_pokemon))
            moves.append(self.emb._embed_moves_from_pokemon(t2_pokemon))

        num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
        # TODO: edit probs
        pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
        moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))

        x = torch.cat((pokemon, moves.flatten())).unsqueeze(0)
        
        x, hidden = self.rnn(x, hidden)
        
        x = self.softmax(x)

        return x, hidden


def train(data): # data is a dict of list of battles and tensors
    delphox = Delphox(LSTM_INPUT_SIZE, LSTM_OUTPUT_SIZE)

    optimizer = torch.optim.Adam(delphox.parameters(), lr=0.001)

    for battle, tensors in data.items():

        loss = 0
        hidden = (torch.randn(2, LSTM_OUTPUT_SIZE) , torch.randn(2, LSTM_OUTPUT_SIZE))
        team1_history, team2_history = delphox.emb.get_team_histories(battle)

        for idx, ((team1, team2), tensor) in enumerate(zip(zip(team1_history, team2_history), tensors)):

            my_active, opponent_active = battle[idx].active_pokemon, battle[idx].opponent_active_pokemon

            my_moves = POKEDEX[my_active.species]['moves']
            opponent_moves = POKEDEX[opponent_active.species]['moves']

            non_zeros = []
            for m in my_moves:
                non_zeros.append(MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)
            
            for m in opponent_moves:
                non_zeros.append((TOTAL_POSSIBLE_MOVES + 1) + MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)
            
            #print(non_zeros)
            non_zeros = torch.tensor(non_zeros, dtype = torch.int64)
            
            output, hidden = delphox(team1, team2, hidden)

            output = output.squeeze(0)
            
            mask = torch.zeros_like(output)
            mask.scatter_(0, non_zeros, 1)

            output = torch.mul(output, mask)

            output = delphox.softmax(output)

            loss += delphox.loss(output, tensor)

        optimizer.zero_grad()
        loss.backward()
        print(f"{loss=}")
        optimizer.step()


if __name__ == "__main__":
    train(SMALL_DATASET)