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
        self.unknown_value = -1

    def extract_side_conditions(self, curr_turn: Battle) -> torch.Tensor:
        indices = {side_condition.name: idx for idx, side_condition in enumerate(SideCondition)}

        side_conditions = curr_turn._side_conditions
        other_side_conditions = curr_turn._opponent_side_conditions

        res = torch.Tensor([0] * len(indices) * 2)

        for sc, amt in side_conditions.items():
            print(sc.name, sc.value)
            res[indices[sc.name]] = amt
        
        for sc, amt in other_side_conditions.items():
            res[indices[sc.name] + len(indices)] = amt
        
        return res

    def extract_weather(self, curr_turn: Battle) -> torch.Tensor:
        indices = {weather.name: idx for idx, weather in enumerate(Weather)}

        weather = curr_turn.weather

        res = torch.Tensor([0] * len(indices))

        if weather:
            res[indices[weather.name]] = 1
        
        return res

    def extract_pokemon(self, curr_turn: Battle) -> torch.Tensor:

        def extract_base_stats(pokemon: Pokemon) -> torch.Tensor:
            stats = torch.zeros(6)
            for idx, value in enumerate(pokemon.base_stats.values()):
                stats[idx] = value
            
            return stats
        def extract_boosts(pokemon: Pokemon) -> torch.Tensor:
            boosts = torch.zeros(7)
            for idx, value in enumerate(pokemon.boosts.values()):
                boosts[idx] = value
            return boosts
        
        def extract_types(pokemon: Pokemon) -> torch.Tensor:
            type = torch.zeros(2)
            # print(pokemon.types)
            type[0] = pokemon.types[0].value
            type[1] = self.unknown_value if pokemon.types[1] == None else pokemon.types[1].value
            return type

        def extract_status(pokemon: Pokemon) -> torch.Tensor:
            statuses = torch.zeros(7)
            indices = {status.name: idx for idx, status in enumerate(Status)}

            if pokemon.status is not None:
                statuses[indices[pokemon.status.name]] = 1

            return statuses
        
        res = torch.zeros(12, 6 + 7 + 2 + 7)

    
        for i, pokemon in enumerate(curr_turn.team.values()):
            if pokemon._status == Status.BRN:
                pokemon._base_stats['atk'] = pokemon._base_stats['atk'] // 2
            elif pokemon._status == Status.PAR:
                pokemon._base_stats['spe'] = pokemon._base_stats['spe'] // 2
            res[i] = torch.cat([extract_base_stats(pokemon), extract_boosts(pokemon), extract_types(pokemon), extract_status(pokemon)])
        
        if len(curr_turn.team) < 6:
            for i in range(len(curr_turn.team), 6):
                res[i] = torch.Tensor([self.unknown_value] * 22)
        
        for i, pokemon in enumerate(curr_turn.opponent_team.values(), start=6):
            if pokemon.status == Status.BRN:
                pokemon._base_stats['atk'] = pokemon._base_stats['atk'] // 2
            elif pokemon.status == Status.PAR:
                pokemon._base_stats['spe'] = pokemon._base_stats['spe'] // 2
                
            res[i] = torch.cat([extract_base_stats(pokemon), extract_boosts(pokemon), extract_types(pokemon), extract_status(pokemon)])

        if len(curr_turn.opponent_team) < 6:
            for i in range(len(curr_turn.opponent_team), 6):
                res[i + 6] = torch.Tensor([self.unknown_value] * 22)
        
        return res

    def extract_field(self, curr_turn: Battle) -> torch.Tensor:
        indices = {field.name: idx for idx, field in enumerate(Field)}

        field = curr_turn.fields

        res = torch.zeros(len(indices))

        if field:
            res[indices[field.name]] = 1
        
        return res

    def extract_effects(self, curr_turn: Battle) -> torch.Tensor:
        num_effects = 0
        indices = {}

        for idx, effect in enumerate(Effect):
            indices[effect.name] = idx
            num_effects += 1
        
        res = torch.zeros(12, num_effects)

        for i, pokemon in enumerate(curr_turn.team.values()):
            for effect in pokemon.effects:
                res[i][indices[effect.name]] = 1
        if len(curr_turn.team) < 6:
            for i in range(len(curr_turn.team), 6):
                res[i] = torch.Tensor([self.unknown_value] * num_effects)
        
        for i, pokemon in enumerate(curr_turn.opponent_team.values(), start=6):
            for effect in pokemon.effects:
                res[i][indices[effect.name]] = 1
        if len(curr_turn.opponent_team) < 6:
            for i in range(len(curr_turn.opponent_team), 6):
                res[i + 6] = torch.Tensor([self.unknown_value] * num_effects)

    def extract_moves(self, curr_turn: Battle) -> torch.Tensor:
        # move while contain base power (1 element), status chances(1 probability value 0<x<1 for each status), accuracy (1 element), type (1 for the appropriate type, 0 other wise)\
        # expected hits, weather, and hazard
        status_indices = {status.name: idx for idx, status in enumerate(Status)}
        type_indices = {type.name: idx for idx, type in enumerate(PokemonType)}
        weather_indices = {weather.name: idx for idx, weather in enumerate(Weather)}
        sidecondition_indices = {sidecondition.name: idx for idx, sidecondition in enumerate(SideCondition)}

        sc_str_to_enum = {sc.name.replace('_', '').lower(): sc for sc in SideCondition}

        res = torch.zeros(12, 4, 1 + len(status_indices) + 1 + len(type_indices) + 1 + len(weather_indices) + len(sidecondition_indices))
        for i, pokemon in enumerate(curr_turn.team.values()):
            for j, move in enumerate(pokemon.moves.values()):

                res[i][j][0] = move.base_power

                if move.status is not None:
                    res[i][j][1 + status_indices[move.status.name]] = 1

                res[i][j][1 + len(status_indices)] = move.accuracy

                
                res[i][j][2 + len(status_indices) + type_indices[move.type.name]] = 1

                res[i][j][2 + len(status_indices) + len(type_indices)] = move.expected_hits

                if move.weather is not None:
                    res[i][j][3 + len(status_indices) + len(type_indices) + weather_indices[move.weather.name]] = 1

                if move.side_condition is not None:
                    sc = move.side_condition.split(' ')[0].replace('_', '').lower()
                    res[i][j][3 + len(status_indices) + + len(type_indices) + len(weather_indices) + sidecondition_indices[sc_str_to_enum[sc].name]] = 1

            if len(pokemon.moves) < 4:
                for k in range(len(pokemon.moves), 4):
                    res[i][k] = torch.Tensor([self.unknown_value] * (1 + len(status_indices) + 1 + len(type_indices) + 1 + len(weather_indices) + len(sidecondition_indices)))
                #res[i][j] = torch.Tensor([self.unknown_value] * (1 + len(status_indices) + 1 + len(type_indices) + 1 + len(weather_indices) + len(sidecondition_indices)))

        if len(curr_turn.team) < 6:
            for i in range(len(curr_turn.team), 6):
                res[i] = torch.Tensor([self.unknown_value] * 4 * (1 + len(status_indices) + 1 + len(type_indices) + 1 + len(weather_indices) + len(sidecondition_indices))).reshape(4, -1)
        
        for i, pokemon in enumerate(curr_turn.opponent_team.values(), start=6):
            for j, move in enumerate(pokemon.moves.values()):
            
                res[i][j][0] = move.base_power

                if move.status is not None:
                    res[i][j][1 + status_indices[move.status.name]] = 1

                res[i][j][1 + len(status_indices)] = move.accuracy

                res[i][j][2 + len(status_indices) + type_indices[move.type.name]] = 1

                res[i][j][2 + len(status_indices) + len(type_indices)] = move.expected_hits

                if move.weather is not None:
                    res[i][j][3 + len(status_indices) + len(type_indices) + weather_indices[move.weather.name]] = 1

                if move.side_condition is not None:
                    #print(type(move.side_condition))
                    sc = move.side_condition.split(' ')[0].replace('_', '').lower()
                    # print(sc)
                    # print(sc_str_to_enum[sc])
                    res[i][j][3 + len(status_indices) + + len(type_indices) + len(weather_indices) + sidecondition_indices[sc_str_to_enum[sc].name]] = 1

            if len(pokemon.moves) < 4:
                for k in range(len(pokemon.moves), 4):
                    res[i][k] = torch.Tensor([self.unknown_value] * (1 + len(status_indices) + 1 + len(type_indices) + 1 + len(weather_indices) + len(sidecondition_indices)))
        
        if len(curr_turn.team) < 6:
                for i in range(len(curr_turn.team), 6):
                    res[i + 6] = torch.Tensor([self.unknown_value] * 4 * (1 + len(status_indices) + 1 + len(type_indices) + 1 + len(weather_indices) + len(sidecondition_indices))).reshape(4, -1)

        return res
        
    
    def embed(self, curr_turn: Battle) -> torch.embedding:
        """
        Calls all extraction functions and concatenates the results into a single embedding

        Input: Battle object representing a turn
        Output: torch embedding of the turn
        """
        
        hazards = self.extract_side_conditions(curr_turn)
        pokemon = self.extract_pokemon(curr_turn)
        moves = self.extract_moves(curr_turn)
        field = self.extract_field(curr_turn)
        weather = self.extract_weather(curr_turn)
        effects = self.extract_effects(curr_turn)

        print("successfully extracted all features")


    def process_battle(self) -> list[Battle]:
        with open(self.battle_json) as f:
            battle_data = json.load(f)
        
        history = battle_data['log']

        history = history.split('\n')

        # initialize logger
        logger = logging.getLogger('poke-env')

        curr_battle = Battle(battle_data['id'], battle_data['p1'],logger, 8)

        curr_battle._opponent_username = battle_data['p2']

        battle_objects = []

        for line in history:
            try:
                curr_battle._parse_message(line.split('|'))
                if line.split('|')[1] == 'turn':
                    battle_objects.append(copy.deepcopy(curr_battle))
            except:
                continue
        
        return battle_objects