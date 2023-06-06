import random
import itertools

import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment import Battle, Pokemon, Move

from .embedder import Embedder
from .constants import POKEDEX, NUM_POKEMON_PER_TEAM, MAX_MOVES
from .calculator import calc_damage

import math
import random
import statistics

EMBEDDER = Embedder()
POSSIBLE_ZOROARK_MOVES = sorted(POKEDEX['zoroark']['moves'].keys())
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


class Victini(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 30, bias=True),
            nn.LogSigmoid(),
            nn.Linear(30, 30, bias=True),
            nn.LogSigmoid(),
            nn.Linear(30, 2, bias=False),
            nn.Softmax(dim=0),
        )
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x.squeeze())

def make_damages(team1: list[Pokemon], team2: list[Pokemon], turn: Battle):
    team1 = team1 + [None] * (NUM_POKEMON_PER_TEAM - len(team1))
    team2 = team2 + [None] * (NUM_POKEMON_PER_TEAM - len(team2))
    damages = []
    unknown_damage = torch.full((MAX_MOVES,), fill_value=-1)
    for mon1, mon2 in itertools.product(team1, team2):
        if mon1 is None or mon2 is None:
            damages.append(unknown_damage)
        else:
            # TODO: handle known moves
            moves = [Move(move, 8) for move in POKEDEX[mon1.species]["moves"]]
            damage = [statistics.mean(calc_damage(mon1, move, mon2, turn)) for move in moves]
            damage = damage + [-1] * (MAX_MOVES - len(damage))
            damages.append(damage)
    return torch.Tensor(damages).flatten()


def make_x(turn: Battle):
    team1 = list(turn.team.values())
    team2 = list(turn.opponent_team.values())
    random.shuffle(team1)
    random.shuffle(team2)
    pokemon1 = [EMBEDDER.embed_pokemon(mon, mon.species == turn.active_pokemon.species) for mon in team1]
    pokemon2 = [EMBEDDER.embed_pokemon(mon, mon.species == turn.opponent_active_pokemon.species) for mon in team2]
    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(pokemon1) - len(pokemon2)
    pokemon = F.pad(torch.hstack(pokemon1 + pokemon2), (0, num_unknown_pokemon * pokemon1[0].shape[0]), mode='constant', value=-1)
    field_conditions = EMBEDDER.embed_conditions(turn, opponent_pov=False)
    damages = make_damages(team1, team2, turn)
    x = torch.cat((pokemon, field_conditions, damages)).unsqueeze(0)
    return x

def train(victini: Victini, data, discount=0.5, lr=0.01, weight_decay=1e-5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.SGD(victini.model.parameters(), lr=lr, weight_decay=weight_decay)
    win_predicts = 0
    loss_predicts = 0

    new_data = []
    for turns, won in data:
        for i, turn in enumerate(turns):
            gamma = 1 - discount / math.exp(i)
            new_data.append((turn, won, gamma))

    random.shuffle(new_data)

    num_correct = 0
    num_wrong = 0
    for turn, won, gamma in new_data:
        y_true = torch.Tensor([1, 0]) if won else torch.Tensor([0, 1])
        optimizer.zero_grad()
        x = make_x(turn)
        y_pred = victini(x)
        print(y_pred)
        cross_entropy_loss = victini.loss(y_pred, y_true)
        L = gamma * cross_entropy_loss
        L.backward()

        optimizer.step()
        if y_pred.argmax().item() == 0:
            win_predicts += 1
        else:
            loss_predicts += 1
        if (y_pred.argmax().item() == 0 and won) or (y_pred.argmax().item() == 1 and not won):
            print(GREEN + f"loss: {L.item()}" + RESET)
            num_correct += 1
        else:
            print(RED + f"loss: {L.item()}" + RESET)
            num_wrong += 1
    print(f"###\n"
          f"overall accuracy:\t{num_correct / (num_wrong + num_correct + 1e-10)}\n"
          f"win predicts:\t{win_predicts}\n"
          f"loss predicts:\t{loss_predicts}\n"
          f"###")


def evaluate(victini: Victini, data):
    total_wrong = 0
    total_correct = 0
    win_predicts = 0
    loss_predicts = 0
    for turns, won in data:
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag} ###")
        num_correct = 0
        num_wrong = 0
        for i, turn in enumerate(turns):
            x = make_x(turn)
            y_pred = victini(x)
            print(y_pred)
            if y_pred.argmax().item() == 0:
                win_predicts += 1
            else:
                loss_predicts += 1
            if (y_pred.argmax().item() == 0 and won) or (y_pred.argmax().item() == 1 and not won):
                print(GREEN + "correct" + RESET)
                num_correct += 1
            else:
                print(RED + "wrong" + RESET)
                num_wrong += 1
        total_wrong += num_wrong
        total_correct += num_correct
        print(f"###\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"win predicts:\t{win_predicts}\n"
              f"loss predicts:\t{loss_predicts}\n"
              f"###")

