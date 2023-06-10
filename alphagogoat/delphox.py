import random
import itertools
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment import Battle, Pokemon, Move, Status

from .constants import POKEDEX, NUM_POKEMON_PER_TEAM, MAX_MOVES
from .embedder import Embedder
from .utils import vec2action, action2vec
from .calculator import calc_damage

import math
import random

EMBEDDER = Embedder()
POSSIBLE_ZOROARK_MOVES = sorted(POKEDEX['zoroark']['moves'].keys())
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


class Delphox(nn.Module):
    def __init__(self, input_size, hidden_size=300):
        super().__init__()
        # TODO: test LSTM
        self.l1 = nn.Linear(input_size, hidden_size, bias=True)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.a2 = nn.Sigmoid()
        self.l3 = nn.Linear(hidden_size, MAX_MOVES + NUM_POKEMON_PER_TEAM, bias=False)
        # self.softmax = nn.Softmax(dim=0)
        self.softmax = nn.Sigmoid()

        self.loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, mask):
        y = self.l1(x.squeeze(0))
        y = self.a1(y)
        y = self.l2(y)
        y = self.a2(y)
        y = self.l3(y)
        y = torch.where(~mask, torch.tensor(-1e10), y)
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

def get_mask(pokemon: Pokemon):
    # TODO: toggle moves that we know aren't there for sure
    mask = torch.zeros(MAX_MOVES + 1, dtype=bool)
    mask[len(POKEDEX[pokemon.species]["moves"]):-1] = True
    return mask


def get_legality(turn: Battle, opponent_pov: bool):
    # TODO: ask Adam if this breaks gradient prop
    if opponent_pov:
        pokemon = turn.opponent_active_pokemon
        team = turn.opponent_team
    else:
        pokemon = turn.active_pokemon
        team = turn.team

    # True = legal action, False = legal action
    legal = torch.zeros(MAX_MOVES + NUM_POKEMON_PER_TEAM, dtype=bool)


    # MOVES
    # start by assuming all moves are illegal

    # if we know the moveset, only unmask those (if PP > 0)
    if len(pokemon.moves) == 4:
        # NOTE: poke-env doesn't store dynamax moves in pokemon.moves, so we don't have to worry about them
        legal[:MAX_MOVES] = False
        moves = sorted(POKEDEX[pokemon.species]["moves"])
        for move in pokemon.moves.values():
            # TODO: dynamaxed moves messes up the pp thing (make sure to fix that)
            if move.current_pp > 0:
                continue
            i = moves.index(move.id)
            legal[i] = True
    # otherwise, only unmask the number of moves possible for that pokemon
    else:
        legal[:len(POKEDEX[pokemon.species]["moves"])] = True


    # SWITCHES
    # start by assuming all switches are illegal

    # legalize all unfainted (not active) pokemon
    team_species = [mon.species for mon in team.values()]
    for mon in team.values():
        if mon.status != Status.FNT and mon.species != pokemon.species:
            i = team_species.index(mon.species)
            legal[MAX_MOVES + i] = True

    # also legalize an unseen pokemon if there are still unknown pokemon in the team (i.e., len(team) < 6)
    if len(team) < 6:
        legal[MAX_MOVES + len(team)] = 1

    return legal


def process_data(data: list[list[Battle], list[tuple[str, str], list[tuple[str, str]]]], discount: float):
    # unpack examples
    examples = []
    for turns, actions1, actions2 in data:
        for i, (turn, a1, a2) in enumerate(zip(turns, actions1, actions2)):
            gamma = 1 - discount / math.exp(i)
            v1 = action2vec(a1, turn.team, turn.active_pokemon)
            v2 = action2vec(a2, turn.opponent_team, turn.opponent_active_pokemon)
            examples += [(turn, a1, v1, gamma, False), (turn, a2, v2, gamma, True)]

    # resample for an equal number of moves and switches
    moves = [ex for ex in examples if ex[1][1] == 'move']
    switches = [ex for ex in examples if ex[1][1] == 'switch']
    length = max(len(moves), len(switches))
    moves = random.choices(moves, k=length)
    switches = random.choices(switches, k=length)
    new_data = itertools.chain(*zip(moves, switches))

    # # sort examples by the index of the correct move
    # indexed_examples = {i: [] for i in range(MAX_MOVES + NUM_POKEMON_PER_TEAM)}
    # for ex in examples:
    #     i = ex[2].argmax().item()
    #     indexed_examples[i].append(ex)
    #
    # # resample and shuffle examples
    # length = max([len(li) for li in indexed_examples.values() if li])
    # resampled_examples = []
    # for li in indexed_examples.values():
    #     if not li:
    #         continue
    #     try:
    #         resampled = random.choices(li, k=length)
    #     except IndexError:
    #         resampled = li
    #     random.shuffle(resampled)
    #     resampled_examples.append(resampled)
    #
    # interleave resampled examples to make new data
    # new_data = itertools.chain(*zip(*resampled_examples))
    return new_data


def train(delphox: Delphox, data, lr=0.001, discount=0.5, weight_decay=1e-5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.SGD(delphox.parameters(), lr=lr, weight_decay=weight_decay)
    total_wrong = 0
    total_correct = 0

    new_data = process_data(data, discount)

    for turn, action, vec, gamma, opponent_pov in new_data:
        optimizer.zero_grad()
        x = make_x(turn, opponent_pov)
        mask = get_legality(turn, opponent_pov)
        print(turn.battle_tag)
        print(mask)
        pred = delphox(x, mask)
        pred_action = vec2action(pred, turn, opponent_pov)

        if action[0] == 'switch':
            team = turn.opponent_team.values() if opponent_pov else turn.team.values()
            team = [mon.species for mon in team]
            switched_in = action[1]
            if not any(species.startswith(switched_in) for species in team):
                action = ("switch", f"unseen {action[1]}")

        if pred.argmax().item() == vec.argmax().item():
            total_correct += 1
            color = GREEN
        else:
            total_wrong += 1
            color = RED

        if opponent_pov:
            mon2 = turn.active_pokemon.species
            mon1 = turn.opponent_active_pokemon.species
        else:
            mon1 = turn.active_pokemon.species
            mon2 = turn.opponent_active_pokemon.species

        print(color + "{:<30} {:<30} {:<30} {:<30}".format(mon1, mon2, str(pred_action), str(action)) + RESET)
        print(np.around(pred.detach().numpy(), decimals=3))
        L = gamma * (delphox.loss(pred, vec))
        print(f"loss: {L.item()}")
        L.backward()
        optimizer.step()

    print(f"###\n"
          f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
          f"###")