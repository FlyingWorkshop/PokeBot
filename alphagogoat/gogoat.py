import math
import random

import numpy as np
import torch
from gym.spaces import Space, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.battle import Battle
import delphox
from constants import LSTM_INPUT_SIZE
from pokedex import POKEDEX
import torch.nn as nn
import numpy as np
from embedder import Embedder

EMB = Embedder() 

class Node:
    def __init__(self, state: Battle):
        self.state = state
        self.children = []
        self.parent = None
        self.visit_count = 0
        self.value = 0
        self.seer = delphox.Delphox(LSTM_INPUT_SIZE)
    
    def expand(self, action_space):
        for action in action_space:
            # Assuming that `state_transition_function` is a function
            # that takes a state and an action and returns a new state.
            new_state = state_transition_function(self.state, action)
            child = Node(new_state)
            child.parent = self
            self.children.append(child) # should we explore more than one child? or just best child?

# Delphox is our policy_net? Or value net? Probably policynet

#delphox reutrns a move tensor; we can 
class MCTS(nn.Module):
    def __init__(self, q_net: nn.Module):
        self.q_net = q_net

    def selection(self, node: Node):
        
        possible_actions = [p for p in POKEDEX[node.state.active_pokemon]['moves'].keys() if p != 'struggle'] + ['switch']
        possible_actions = [EMB.embed_move(p) for p in possible_actions]

        return possible_actions
    
    def simulation(self, node: Node):
        # Use the value network to estimate the game outcome
        state_tensor = torch.tensor(node.state).unsqueeze(0)
        with torch.no_grad():
            value_estimate = self.value_net(state_tensor)

        # Convert the value estimate to a simple scalar
        value_estimate = value_estimate.item()

        return value_estimate

    def backpropagation(self, node: Node, result):
        # Implement backpropagation phase here
        pass

    def expansion(self, node: Node):
        # Implement expansion phase here
        pass


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