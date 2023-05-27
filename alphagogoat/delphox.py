import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle
from tqdm.auto import tqdm

from .embedder import Embedder, get_team_histories
from .constants import *

EMBEDDER = Embedder()

class Delphox(nn.Module):
    LSTM_OUTPUT_SIZE = len(MoveEnum) + 1

    def __init__(self, input_size, hidden_layers=2):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        # TODO: maybe add an encoder?
        self.rnn = nn.LSTM(input_size, Delphox.LSTM_OUTPUT_SIZE, hidden_layers)
        self.softmax = nn.Softmax(dim=0)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        move, hidden = self.rnn(x, hidden)
        move = self.softmax(move)
        return move, hidden

    # def forward(self, team1: dict[str: Pokemon], team2: dict[str: Pokemon], hidden: tuple[torch.tensor]):
    #     pokemon = []
    #     moves = []
    #
    #     for t1_pokemon in team1.values():
    #         pokemon.append(self.emb.embed_pokemon(t1_pokemon).to(device=device))
    #         moves.append(self.emb.embed_moves_from_pokemon(t1_pokemon).to(device=device))
    #
    #     for t2_pokemon in team2.values():
    #         pokemon.append(self.emb.embed_pokemon(t2_pokemon).to(device=device))
    #         moves.append(self.emb.embed_moves_from_pokemon(t2_pokemon).to(device=device))
    #
    #     num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    #     pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    #     moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))
    #     x = torch.cat((pokemon, moves.flatten())).unsqueeze(0)
    #     x, hidden = self.rnn(x, hidden)
    #     x = self.softmax(x)
    #     return x, hidden

def make_x(turn: Battle, team1: dict[str: Pokemon], team2: dict[str: Pokemon]):
    pokemon = []
    moves = []

    for t1_pokemon in team1.values():
        pokemon.append(EMBEDDER.embed_pokemon(t1_pokemon).to(device=device))
        moves.append(EMBEDDER.embed_moves_from_pokemon(t1_pokemon).to(device=device))

    for t2_pokemon in team2.values():
        pokemon.append(EMBEDDER.embed_pokemon(t2_pokemon).to(device=device))
        moves.append(EMBEDDER.embed_moves_from_pokemon(t2_pokemon).to(device=device))

    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))
    x = torch.cat((pokemon, moves.flatten())).unsqueeze(0)
    return x


def train(delphox: Delphox, data, lr=0.001, discount=0.5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.Adam(delphox.parameters(), lr=lr)
    for turns, history1, history2, moves1, moves2 in data:
        # TODO: punish bad predictions on early turns less
        # TODO: have representations of the future
        for i, (turn, team1, team2, move1, move2) in tqdm(enumerate(zip(turns, history1, history2, moves1, moves2)), total=len(turns)):
            gamma = 1 - discount / torch.exp(i)
            x1 = make_x(turn, team1, team2)
            move1_pred = delphox(x1)

            optimizer.zero_grad()
            loss = gamma * delphox.loss(move1_pred, move1)
            loss.backward()
            optimizer.step()

            x2 = make_x(turn, team2, team1)
            move2_pred = delphox(x2)
            optimizer.zero_grad()
            loss = gamma * delphox.loss(move2_pred, move1)
            loss.backward()
            optimizer.step()


    # for _ in range(reps):
    #     for battle, h1, h2, tensors_grid in tqdm(data):
    #         loss = 0
    #         hidden = (torch.randn(2, LSTM_OUTPUT_SIZE).to(device=device) , torch.randn(2, LSTM_OUTPUT_SIZE).to(device=device))
    #         for turn, team1, team2, tensor in zip(battle, h1, h2, tensors_grid):
    #             pokemon1, pokemon2 = [], []
    #             moves1, moves2 = [], []
    #
    #             for t1_pokemon in team1.values():
    #                 pokemon1.append(delphox.emb.embed_pokemon(t1_pokemon).to(device=device))
    #                 moves2.append(delphox.emb.embed_moves_from_pokemon(t1_pokemon).to(device=device))
    #
    #             for t2_pokemon in team2.values():
    #                 pokemon1.append(delphox.emb.embed_pokemon(t2_pokemon).to(device=device))
    #                 moves2.append(delphox.emb.embed_moves_from_pokemon(t2_pokemon).to(device=device))
    #
    #             tensor = tensor.to(device=device)
    #             my_active, opponent_active = turn.active_pokemon, turn.opponent_active_pokemon
    #
    #             my_moves = {} if my_active.species == 'typenull' else POKEDEX[my_active.species]['moves']
    #             opponent_moves = {} if opponent_active.species == 'typenull' else POKEDEX[opponent_active.species]['moves']
    #
    #             non_zeros = []
    #             for m in my_moves:
    #                 non_zeros.append(MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)
    #
    #             for m in opponent_moves:
    #                 non_zeros.append((TOTAL_POSSIBLE_MOVES + 1) + MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)
    #
    #             non_zeros = torch.tensor(non_zeros, dtype=torch.int64).to(device=device)
    #
    #             for x in [a, b]:
    #             e
    #             output = output.squeeze(0)
    #             mask = torch.zeros_like(output).to(device=device)
    #             mask.scatter_(0, non_zeros, 1)
    #             output = torch.mul(output, mask)
    #             output = delphox.softmax(output)
    #             loss += delphox.loss(output, tensor)
    #
    #         optimizer.zero_grad()
    #         # TODO: fix this hacky solution on example 103/145
    #         if isinstance(loss, torch.Tensor):
    #             loss.backward()
    #         print(f"### {loss=}")
    #         optimizer.step()

# if __name__ == "__main__":
#     train(SMALL_DATASET)
