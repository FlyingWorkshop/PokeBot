import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.abstract_battle import AbstractBattle
import math
import random
import torch



class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self._number_of_visits = 0
        self._total_reward = 0

    def number_of_visits(self):
        return self._number_of_visits

    def total_reward(self):
        return self._total_reward

    def is_fully_expanded(self):
        return len(self.untried_actions()) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            c.total_reward() / (c.number_of_visits()) + c_param * math.sqrt((2 * math.log(self.number_of_visits()) / (c.number_of_visits())))
             for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def rollout_policy(self, possible_moves):        
        return possible_moves[random.randint(0, len(possible_moves) - 1)]

    def expand(self):
        action = self.untried_actions().pop()
        next_state = self.state.move(action)
        child_node = Node(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def rollout_policy(self, possible_moves, net):
        # Convert your state to a tensor
        state_tensor = torch.tensor(self.state).float()
        # Pass state through the network
        action_probabilities = net(state_tensor)
        # Select action based on the output of your network
        action = torch.argmax(action_probabilities).item()
        return action
    
    def backpropagate(self, reward):
        self._number_of_visits += 1.
        self._total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTS:
    def __init__(self, root):
        self.root = root

    def search(self, iterations):
        for _ in range(iterations):
            node = self._tree_policy()
            reward = node.rollout()
            node.backpropagate(reward)
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        current_node = self.root
        while not current_node.state.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node



class SimpleRLPlayer(Gen8EnvSinglePlayer):
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