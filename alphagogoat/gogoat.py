import math
import random

import numpy as np
import torch
from gym.spaces import Space, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status
from .delphox import Delphox
from .delphox import make_x as delphox_make_x
from .delphox import get_legality as delphox_make_mask
from .victini import Victini
from .constants import LSTM_INPUT_SIZE, MoveEnum
from .pokedex import POKEDEX
import torch.nn as nn
import numpy as np
from .embedder import Embedder
from itertools import product
import math
from .victini import computeFuture

EMB = Embedder() 

VICT = Victini(940)
VICT.load_state_dict(torch.load("victini_gogoat.pth"))

DELPH = Delphox(2825)
DELPH.load_state_dict(torch.load("delphox_gogoat.pth"))

class MCTS:
    def __init__(self):
        self.delphox = DELPH
        self.victini = VICT
        self.future = computeFuture

    def selectExpand(self, node: Battle):

        if node.team.values():
            possible_switches_me = [p for p in node.team.values() if p != node.active_pokemon and p.status == Status.FNT]
        num_fainted_me = 0
        for p in node.team.values():
            if p.status == Status.FNT:
                num_fainted_me += 1
        if len(possible_switches_me) + num_fainted_me < 5:
            possible_switches_me.append("unknown")
        
        if node.opponent_team.values():
            possible_switches_opponent = [p for p in node.opponent_team.values() if p != node.opponent_active_pokemon and p.status == Status.FNT]
        num_fainted_opponent = 0
        for p in node.opponent_team.values():
            if p.status == Status.FNT:
                num_fainted_opponent += 1
        if len(possible_switches_opponent) +  num_fainted_opponent < 5:
            possible_switches_opponent.append("unknown")
        
        possible_actions_me = list(product([node.active_pokemon.species], [p for p in POKEDEX[node.active_pokemon.species]['moves'].keys() if p != 'struggle'])) + list(product(['switch'], possible_switches_me))
        possible_actions_opponent = list(product([node.opponent_active_pokemon.species], [p for p in POKEDEX[node.opponent_active_pokemon.species]['moves'].keys() if p != 'struggle'])) + list(product(['switch'], possible_switches_opponent))

        return list(set(product(possible_actions_me, possible_actions_opponent)))
    
    def simulation(self, node: Battle, action) -> float:
        # Use the value network to estimate the game outcome
        delphox_input, delphox_mask = delphox_make_x(node, False), delphox_make_mask(node, False)
        

        with torch.no_grad():
            dist = self.delphox(delphox_input, delphox_mask)
            prob1, prob2 = dist[MoveEnum[action[0]] - 1], dist[MoveEnum[action[1] - 1]]
            future = self.future(node, action)
            victini_input = future
            value_estimate = self.victini(victini_input)
            value_estimate = value_estimate[0] - value_estimate[1]
        # Convert the value estimate to a simple scalar
        return prob1 * prob2 * value_estimate

    def get_action(self, node: Battle) -> str:
        action_pairs = self.selectExpand(node)
        best_action = None
        best_score = -math.inf
        for action in action_pairs:
            score = self.simulation(node, action)
            if score > best_score:
                best_score = score
                best_action = action[0]
                if best_action == 'switch':
                    best_action += ' ' + action[1]
        
        if not best_action.startswith('switch'):
            return Move(best_action, gen=8)
    
        if not best_action.endswith('unknown'):
            return Pokemon(species=best_action.split(' ')[1], gen=8)
        else:
            return 'random'