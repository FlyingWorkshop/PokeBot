import random
import itertools

import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment import Battle, Pokemon, Move, SideCondition
from poke_env.environment.status import Status
from poke_env.environment.side_condition import STACKABLE_CONDITIONS

from .embedder import Embedder
from .constants import POKEDEX, NUM_POKEMON_PER_TEAM, MAX_MOVES
from .calculator import calc_damage

import math
import random
import statistics

from copy import deepcopy

EMBEDDER = Embedder()
POSSIBLE_ZOROARK_MOVES = sorted(POKEDEX['zoroark']['moves'].keys())
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
UNKNOWN_POKEMON = Pokemon(gen = 8, species= "suicune")

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

def non_dmg_changes(current: Battle, move: Move, is_opponent: bool) -> None:
    
    if not is_opponent:
        if move.boosts is not None:
            for b in move.boosts:
                if b in current.active_pokemon.boosts:
                    current.opponent_active_pokemon.boosts[b] += move.boosts[b]
                else:
                    current.opponent_active_pokemon.boosts[b] = move.boosts[b]
            
        if move.self_boost is not None:
            for b in move.self_boost:
                if b in current.active_pokemon.boosts:
                    current.active_pokemon.boosts[b] += move.self_boosts[b]
                else:
                    current.active_pokemon.boosts[b] = move.self_boosts[b]
        
        current.active_pokemon._current_hp += move.heal

        if move.side_condition is not None:
            current.side_conditions[SideCondition[move.side_condition]] += 1 if (move.side_condition in STACKABLE_CONDITIONS) else 1
        
        if move.weather is not None:
            current.weather = {move.weather: current.turn}

        if move.terrain is not None:
            current.fields = current._field_start(move.terrain)
    else:
        if move.boosts is not None:
            for b in move.boosts:
                if b in current.opponent_active_pokemon.boosts:
                    current.active_pokemon.boosts[b] += move.boosts[b]
                else:
                    current.active_pokemon.boosts[b] = move.boosts[b]

        if move.self_boost is not None:
            for b in move.self_boosts:
                if b in current.active_pokemon.boosts:
                    current.opponent_active_pokemon.boosts[b] += move.self_boosts[b]
                else:
                    current.opponent_active_pokemon.boosts[b] = move.self_boosts[b]
        
        current.opponent_active_pokemon._current_hp += move.heal

        if move.side_condition is not None:
            current.opponent_side_conditions[SideCondition[move.side_condition]] += 1 if (move.side_condition in STACKABLE_CONDITIONS) else 1
        
        if move.weather is not None:
            current.weather = {move.weather: current.turn}

        if move.terrain is not None:
            current.fields = current._field_start(move.terrain)

def computeFuture(current: Battle, action) -> torch.Tensor:
    """
    Takes in a current battle state as well as a proposed tuple of tensor of actions, where the second tensor is the opponent action and the first tensor is the agent action.
    Returns the potential future state after the actions are taken, as a tensor, using make_x from the Victini class
    """
    temp_curr = deepcopy(current)
    if action[0][0] == "switch" and action[1][0] == "switch":

        if action[0][1] == 'unknown' and action[1][1] == 'unknown':
            new_pokemon_me = deepcopy(UNKNOWN_POKEMON)
            new_pokemon_them = deepcopy(UNKNOWN_POKEMON)
        elif action[0][1] != 'unknown' and action[1][1] == 'unknown':
            new_pokemon_me = temp_curr.team[action[0][1]]
            new_pokemon_them = deepcopy(UNKNOWN_POKEMON)
        elif action[0][1] == 'unknown' and action[1][1] != 'unknown':
            new_pokemon_me = deepcopy(UNKNOWN_POKEMON)
            new_pokemon_them = temp_curr.opponent_team[action[1][1]]
        else:
            new_pokemon_me = temp_curr.team[action[0][1]]
            new_pokemon_them = temp_curr.opponent_team[action[1][1]]

        temp_curr.active_pokemon._active = False
        temp_curr.opponent_active_pokemon._active = False

        new_pokemon_me._active = True
        new_pokemon_them._active = True

        temp_curr.team[new_pokemon_me.species] = new_pokemon_me
        temp_curr.opponent_team[new_pokemon_them.species] = new_pokemon_them

        return make_x(temp_curr)
    
    elif action[0][0] == "switch" and action[1][0] != "switch":
        temp_curr.active_pokemon._active = False

        if action[0][1] == 'unknown':
            new_pokemon_me = deepcopy(UNKNOWN_POKEMON)
        else:
            new_pokemon_me = temp_curr.team[action[0][1]]
        
        new_pokemon_me._active = True
        temp_curr.team[new_pokemon_me.species] = new_pokemon_me

        # do damage calculation
        dmg_to_me = sum(calc_damage(temp_curr.opponent_active_pokemon, Move(action[1][1], 8), temp_curr.active_pokemon, temp_curr)) // 2
        if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
            temp_curr.active_pokemon._status= Status['FNT']
        else:
            temp_curr.active_pokemon._current_hp -= dmg_to_me
        
        non_dmg_changes(temp_curr, Move(action[1][1], 8), True)
        
        return make_x(temp_curr)
    
    elif action[0][0] != "switch" and action[1][0] == "switch":
        temp_curr.opponent_active_pokemon._active = False

        if action[1][1] == 'unknown':
            new_pokemon_them = deepcopy(UNKNOWN_POKEMON)
        else:
            new_pokemon_them = temp_curr.opponent_team[action[1][1]]
        
        new_pokemon_them._active = True
        temp_curr.opponent_team[new_pokemon_them.species] = new_pokemon_them

        # do damage calculation
        dmg_to_opp = sum(calc_damage(temp_curr.active_pokemon, Move(action[0][1], 8), temp_curr.opponent_active_pokemon, temp_curr)) // 2
        if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
            temp_curr.opponent_active_pokemon._status= Status['FNT']
        else:
            temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
        
        #do all other battle changes
        non_dmg_changes(temp_curr, Move(action[0][1], 8), True)
        
        return make_x(temp_curr)

    elif action[0][0] != "switch" and action[1][0] != "switch":
        priority_me = Move(action[0][1], 8).priority
        priority_opponent = Move(action[1][1], 8).priority

        if priority_me > priority_opponent:
            dmg_to_opp = sum(calc_damage(temp_curr.active_pokemon, Move(action[0][1], 8), temp_curr.opponent_active_pokemon, temp_curr)) // 2
            if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
                temp_curr.opponent_active_pokemon._status= Status['FNT']
            else:
                temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
                non_dmg_changes(temp_curr, Move(action[0][1], 8), False)
                
            if temp_curr.opponent_active_pokemon._status != Status['FNT']:
                dmg_to_me = sum(calc_damage(temp_curr.opponent_active_pokemon, Move(action[1][1], 8), temp_curr.active_pokemon, temp_curr)) // 2
                if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
                    temp_curr.active_pokemon._status= Status['FNT']
                else:
                    temp_curr.active_pokemon._current_hp -= dmg_to_me
                    non_dmg_changes(temp_curr, Move(action[1][1], 8), True)

        elif priority_me < priority_opponent:
            dmg_to_me = sum(calc_damage(temp_curr.opponent_active_pokemon, Move(action[1][1], 8), temp_curr.active_pokemon, temp_curr)) // 2
            if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
                temp_curr.active_pokemon._status= Status['FNT']
            else:
                temp_curr.active_pokemon._current_hp -= dmg_to_me
                non_dmg_changes(temp_curr, Move(action[1][1], 8), True)
        
            if temp_curr.active_pokemon._status != Status['FNT']:
                dmg_to_opp = sum(calc_damage(temp_curr.active_pokemon, Move(action[0][1], 8), temp_curr.opponent_active_pokemon, temp_curr)) // 2
                if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
                    temp_curr.opponent_active_pokemon._status= Status['FNT']
                else:
                    temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
                    non_dmg_changes(temp_curr, Move(action[0][1], 8), False)
        else:
            dmg_to_opp = sum(calc_damage(temp_curr.active_pokemon, Move(action[0][1], 8), temp_curr.opponent_active_pokemon, temp_curr)) // 2
            dmg_to_me = sum(calc_damage(temp_curr.opponent_active_pokemon, Move(action[1][1], 8), temp_curr.active_pokemon, temp_curr)) // 2

            if temp_curr.active_pokemon.base_stats['spe'] > temp_curr.opponent_active_pokemon.base_stats['spe']:
                if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
                    temp_curr.opponent_active_pokemon._status= Status['FNT']
                else:
                    temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
                    non_dmg_changes(temp_curr, Move(action[0][1], 8), False)
                    if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
                        temp_curr.active_pokemon._status= Status['FNT']
                    else:
                        temp_curr.active_pokemon._current_hp -= dmg_to_me
                        non_dmg_changes(temp_curr, Move(action[1][1], 8), True)
            
            elif temp_curr.active_pokemon.base_stats['spe'] < temp_curr.opponent_active_pokemon.base_stats['spe']:
                if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
                    temp_curr.active_pokemon._status= Status['FNT']
                else:
                    temp_curr.active_pokemon._current_hp -= dmg_to_me
                    non_dmg_changes(temp_curr, Move(action[1][1], 8), True)
                    if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
                        temp_curr.opponent_active_pokemon._status= Status['FNT']
                    else:
                        temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
                        non_dmg_changes(temp_curr, Move(action[0][1], 8), False)
                    
            else:
                if random.random() < 0.5:
                    if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
                        temp_curr.opponent_active_pokemon= Status['FNT']
                        temp_curr.opponent_active_pokemon._current_hp = 0
                    else:
                        temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
                        non_dmg_changes(temp_curr, Move(action[0][1], 8), False)
                        if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
                            temp_curr.active_pokemon._status= Status['FNT']
                            temp_curr.active_pokemon._current_hp = 0
                        else:
                            temp_curr.active_pokemon._current_hp -= dmg_to_me
                            non_dmg_changes(temp_curr, Move(action[1][1], 8), True)

                else:
                    if temp_curr.active_pokemon.current_hp - dmg_to_me <= 0:
                        temp_curr.active_pokemon._status= Status['FNT']
                        temp_curr.active_pokemon._current_hp = 0
                    else:
                        temp_curr.active_pokemon._current_hp -= dmg_to_me
                        non_dmg_changes(temp_curr, Move(action[1][1], 8), True)
                        if temp_curr.opponent_active_pokemon.current_hp - dmg_to_opp <= 0:
                            temp_curr.opponent_active_pokemon._status= Status['FNT']
                            temp_curr.opponent_active_pokemon._current_hp = 0
                        else:
                            temp_curr.opponent_active_pokemon._current_hp -= dmg_to_opp
                            non_dmg_changes(temp_curr, Move(action[0][1], 8), False)
        return make_x(temp_curr)

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

