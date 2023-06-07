import random
import itertools
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment import Battle, Pokemon, Move, Status

from .constants import POKEDEX, NUM_POKEMON_PER_TEAM, MAX_MOVES
from .embedder import Embedder
from .utils import vec2str
from .calculator import calc_damage

import math
import random

EMBEDDER = Embedder()
POSSIBLE_ZOROARK_MOVES = sorted(POKEDEX['zoroark']['moves'].keys())
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


class Delphox(nn.Module):
    def __init__(self, input_size, hidden_size=135):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias=True)
        self.a1 = nn.LogSigmoid()
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.a2 = nn.LogSigmoid()
        self.l3 = nn.Linear(hidden_size, MAX_MOVES + NUM_POKEMON_PER_TEAM, bias=False)
        self.softmax = nn.LogSoftmax(dim=0)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, mask):
        y = self.l1(x.squeeze(0))
        y = self.a1(y)
        y = self.l2(y)
        y = self.a2(y)
        y = self.l3(y)
        y = torch.where(mask, torch.tensor(-1e10), y)
        y = self.softmax(y)
        return y

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
            # TODO: should sort?
            moves = [Move(move, 8) for move in sorted(POKEDEX[mon1.species]["moves"])]
            damage = [statistics.mean(calc_damage(mon1, move, mon2, turn)) for move in moves]
            damage = damage + [-1] * (MAX_MOVES - len(damage))
            damages.append(damage)
    return torch.Tensor(damages).flatten()


def make_x(turn: Battle, opponent_pov: bool):
    if opponent_pov:
        team1 = [mon for mon in sorted(turn.team.values(), key=lambda x: x.species)]
        team2 = [mon for mon in sorted(turn.opponent_team.values(), key=lambda x: x.species)]
    else:
        team2 = [mon for mon in sorted(turn.team.values(), key=lambda x: x.species)]
        team1 = [mon for mon in sorted(turn.opponent_team.values(), key=lambda x: x.species)]
    pokemon1 = [EMBEDDER.embed_pokemon(mon, mon.species == turn.active_pokemon.species) for mon in team1]
    pokemon2 = [EMBEDDER.embed_pokemon(mon, mon.species == turn.opponent_active_pokemon.species) for mon in team2]
    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(pokemon1) - len(pokemon2)
    pokemon = F.pad(torch.hstack(pokemon1 + pokemon2), (0, num_unknown_pokemon * pokemon1[0].shape[0]), mode='constant', value=-1)
    field_conditions = EMBEDDER.embed_conditions(turn, opponent_pov=False)
    damages = make_damages(team1, team2, turn)
    x = torch.cat((pokemon, field_conditions, damages)).unsqueeze(0)
    return x


def get_mask(pokemon: Pokemon, team: dict[str: Pokemon]):
    mask = torch.zeros(MAX_MOVES + len(team), dtype=bool)
    mask[len(POKEDEX[pokemon.species]["moves"]):MAX_MOVES] = True
    mask[MAX_MOVES:] = torch.Tensor([mon.status == Status.FNT or mon.species == pokemon.species for mon in sorted(team.values(), key=lambda x: x.species)])
    if len(team) < NUM_POKEMON_PER_TEAM:
        unknown_pokemon = [True for _ in range(NUM_POKEMON_PER_TEAM - len(team))]
        unknown_pokemon[0] = False  # let's Delphox predict switching to an unknown Pokemon
        mask = torch.concat((mask, torch.Tensor(unknown_pokemon)))
    return mask.bool()


def train(delphox: Delphox, data, lr=0.001, discount=0.5, weight_decay=1e-5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.SGD(delphox.parameters(), lr=lr, weight_decay=weight_decay)
    total_wrong = 0
    total_correct = 0

    new_data = []
    for turns, vectors1, vectors2 in data:
        for i, (turn, v1, v2) in enumerate(zip(turns, vectors1, vectors2)):
            gamma = 1 - discount / math.exp(i)
            new_data.append((turn, v1, v2, gamma))

    random.shuffle(new_data)

    for turn, move1, move2, gamma in new_data:
        optimizer.zero_grad()
        x1 = make_x(turn, opponent_pov=False)
        mask1 = get_mask(turn.active_pokemon, turn.team)
        move1_pred = delphox(x1, mask1)
        print(move1_pred)
        L = gamma * (delphox.loss(move1_pred, move1))
        if move1.argmax() == move1_pred.argmax():
            total_correct += 1
            color = GREEN
        else:
            total_wrong += 1
            color = RED

        print(color + "{:<30} {:<30} {:<30} {:<30}".format(
            turn.active_pokemon.species,
            turn.opponent_active_pokemon.species,
            vec2str(move1_pred, turn.active_pokemon, turn.team),
            vec2str(move1, turn.active_pokemon, turn.team)
        ) + RESET)
        print(f"loss: {L.item()}")
        L.backward()

        optimizer.zero_grad()
        x2 = make_x(turn, opponent_pov=True)
        mask2 = get_mask(turn.opponent_active_pokemon, turn.opponent_team)
        move2_pred = delphox(x2, mask2)
        print(move2_pred)
        L = gamma * (delphox.loss(move2_pred, move2))
        if move2.argmax() == move2_pred.argmax():
            total_correct += 1
            color = GREEN
        else:
            total_wrong += 1
            color = RED
        print(color + "{:<30} {:<30} {:<30} {:<30}".format(
            turn.opponent_active_pokemon.species,
            turn.active_pokemon.species,
            vec2str(move2_pred, turn.opponent_active_pokemon, turn.opponent_team),
            vec2str(move2, turn.opponent_active_pokemon, turn.opponent_team)
        ) + RESET)
        print(f"loss: {L.item()}")
        L.backward()
        optimizer.step()


    print(f"###\n"
          f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
          f"###")