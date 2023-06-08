import math
import random

import numpy as np
import torch
from gym.spaces import Space, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.battle import Battle
import delphox
import victini
from constants import LSTM_INPUT_SIZE, MoveEnum
from pokedex import POKEDEX
import torch.nn as nn
import numpy as np
from embedder import Embedder
from itertools import product
import math

EMB = Embedder() 

class MCTS(nn.Module):
    def __init__(self, action_predictor: nn.Module, victory_predictor: nn.Module, future_editor: function):
        self.delphox = action_predictor
        self.victini = victory_predictor
        self.future = future_editor

    def selectExpand(self, node: Battle):
        
        possible_actions_me = [p for p in POKEDEX[node.state.active_pokemon]['moves'].keys() if p != 'struggle'] + ['switch']
        # possible_actions_me = [EMB.embed_move(p) for p in possible_actions_me]

        possible_actions_opponent = [p for p in POKEDEX[node.state.opponent_active_pokemon]['moves'].keys() if p != 'struggle'] + ['switch']
        # possible_actions_opponent = [EMB.embed_move(p) for p in possible_actions_opponent]

        return list(set(product(possible_actions_me, possible_actions_opponent)))
    
    def simulation(self, node: Battle, action: tuple):
        # Use the value network to estimate the game outcome
        delphox_input, delphox_mask = delphox.make_x(node, False), delphox.get_mask(node)
        future = self.future(node, (EMB.embed_move(action[0]), EMB.embed_move(action[1])))
        victini_input = future

        with torch.no_grad():
            value_estimate = self.victini(victini_input)
            value_estimate = value_estimate.item()
            dist = self.delphox(delphox_input, delphox_mask)
            prob1, prob2 = dist[MoveEnum[action[0]] - 1], dist[MoveEnum[action[1] - 1]]

        # Convert the value estimate to a simple scalar
        
        return prob1 * prob2 * value_estimate

    def get_action(self, node: Battle):
        action_pairs = self.selectExpand(node)
        best_action = None
        best_score = -math.inf
        for action in action_pairs:
            score = self.simulation(node, action)
            if score > best_score:
                best_score = score
                best_action = action[0]
        
        return best_action


class AlphaGogoat(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        # TODO: figure out what an ObservationType is
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemon have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )