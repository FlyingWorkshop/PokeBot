import json
import re
from poke_env.environment.abstract_battle import AbstractBattle
import logging
from poke_env.environment.pokemon import Pokemon, PokemonType
import pypokedex
from poke_env.environment.weather import Weather
from poke_env.environment.move import Move
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
import copy
from poke_env.environment.battle import Battle
import torch
from collections import defaultdict
from poke_env.environment.field import Field
from poke_env.environment.effect import Effect

def _get_turns(log) -> list:
    turns = []
    turn = []
    for line in log.split('\n'):
        if line.startswith("|turn|") and turn:
            turns.append(turn)
            turn = []
        turn.append(line)
    return turns
class DataExtractor:
    def __init__(self, battle_json_path: str):
        self.battle_json = battle_json_path
        self.turns = self.process_battle()
        self.unknown_value = 0

    def extract_side_conditions(self, curr_turn: Battle) -> torch.Tensor:
        indices = {side_condition.name: idx for idx, side_condition in enumerate(SideCondition)}

        side_conditions = curr_turn.side_conditions
        other_side_conditions = curr_turn.opponent_side_conditions

        res = torch.Tensor([0] * len(indices) * 2)

        for side_condition, amt in side_conditions.values():
            res[indices[side_condition.name]] = side_conditions[amt]
        
        for side_condition, amt in other_side_conditions.values():
            res[indices[side_condition.name] + len(indices)] = other_side_conditions[amt]
        
        return res

    def extract_weather(self, curr_turn: Battle) -> torch.Tensor:
        indices = {weather.name: idx for idx, weather in enumerate(Weather)}

        weather = curr_turn.weather

        res = torch.Tensor([0] * len(indices))

        if weather:
            res[indices[weather.name]] = 1
        
        return res

    def extract_status(self, curr_turn: Battle) -> torch.Tensor:
        indices = {status.name: idx for idx, status in enumerate(Status)}

        status = curr_turn.status
        opponent_status = curr_turn.opponent_status


    def extract_moves(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_pokemon(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_team(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_types(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_hp(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_boosts(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_field(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_effects(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def embed(self, curr_turn: Battle) -> torch.embedding:
        """
        Calls all extraction functions and concatenates the results into a single embedding

        Input: Battle object representing a turn
        Output: torch embedding of the turn
        """
        pass

    def process_battle(self) -> list[Battle]:
        with open(self.battle_json) as f:
            battle_data = json.load(f)
        
        history = battle_data['log']

        history = history.split('\n')

        # initialize logger
        logger = logging.getLogger('poke-env')

        curr_p1_pokemon = None
        curr_p2_pokemon = None

        curr_battle = Battle(battle_data['id'], battle_data['p1'],logger, 8)

        curr_battle._opponent_username = battle_data['p2']

        battle_objects = []

        for line in history:
            if len(line) <= 1:
                continue
            if line.split('|')[1] == 't:' or line.split('|')[1] == 'win':
                continue
            curr_battle._parse_message(line.split('|'))
            if line.split('|')[1] == 'turn':
                battle_objects.append(copy.deepcopy(curr_battle))
