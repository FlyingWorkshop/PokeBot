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
        self.turns = self.process_battle(battle_json_path)

    def extract_hazards(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_weather(self, curr_turn: Battle) -> torch.Tensor:
        pass

    def extract_status(self, curr_turn: Battle) -> torch.Tensor:
        pass

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

    def embed(self, curr_turn: Battle) -> torch.embedding:
        """
        Calls all extraction functions and concatenates the results into a single embedding

        Input: Battle object representing a turn
        Output: torch embedding of the turn
        """
        pass

    def process_battle(self, battle_log: json) -> list[Battle]:
        """
        input: a json object representing the data of a battle, scraped from pokemon showdown
        output: a list of Battle objects constructed using information from the json

        this function loops through the battle log turn by turn, updating all properties of the Battle. it then appends that object to a list
        
        """

        STACKABLE_CONDITIONS = {SideCondition.SPIKES: 3, SideCondition.TOXIC_SPIKES: 2}

        # initialize logger
        logger = logging.getLogger('poke-env')

        with open(battle_log) as f:
            battle_data = json.load(f)
        
        history = battle_data['log']

        history = history.split('\n')

        curr_p1_pokemon = None
        curr_p2_pokemon = None

        curr_battle = Battle(battle_data['id'], battle_data['p1'],logger, 8)

        generation_dict = {}

        gen1 = set([i for i in range(1, 152)])
        gen2 = set([i for i in range(152, 252)])
        gen3 = set([i for i in range(252, 387)])
        gen4 = set([i for i in range(387, 494)])
        gen5 = set([i for i in range(494, 650)])
        gen6 = set([i for i in range(650, 722)])
        gen7 = set([i for i in range(722, 810)])
        gen8 = set([i for i in range(810, 906)])

        for i in range(1, 906):
            if i in gen1:
                generation_dict[i] = 1
            elif i in gen2:
                generation_dict[i] = 2
            elif i in gen3:
                generation_dict[i] = 3
            elif i in gen4:
                generation_dict[i] = 4
            elif i in gen5:
                generation_dict[i] = 5
            elif i in gen6:
                generation_dict[i] = 6
            elif i in gen7:
                generation_dict[i] = 7
            elif i in gen8:
                generation_dict[i] = 8

        to_skip = []
        battle_log = []

        for idx, line in enumerate(history):
            if len(line) <= 1 or idx in to_skip:
                continue
            
            #initialize some things for the start of a battle, and coniders the two default switches in order to start both the current and opponent team
            if line.startswith("|start"):

                # we have 4 regexes, 2 for each player
                # one for the case with gender, one for the case without
                switch_regex_pattern1 = r"p1a:\s(.*?)\|.*?L(\d+).*?(M|F)\|(\d+)/(\d+)"
                switch_regex_pattern2 = r"p2a:\s(.*?)\|.*?L(\d+).*?(M|F)\|(\d+)/(\d+)"
                genderless_1 = r"p1a:\s(.*?)\|.*?L(\d+).*?(?:M|F)?\|(\d+)/(\d+)"
                genderless_2 = r"p2a:\s(.*?)\|.*?L(\d+).*?(?:M|F)?\|(\d+)/(\d+)"
                p1_info = history[idx + 1]
                p2_info = history[idx + 2]
                to_skip.append(idx + 1)
                to_skip.append(idx + 2)
                match1 = re.search(switch_regex_pattern1, p1_info)
                match2 = re.search(switch_regex_pattern2, p2_info)
                match3 = re.search(genderless_1, p1_info)
                match4 = re.search(genderless_2, p2_info)
                
                if match1:
                    pokemon_name = match1.group(1)
                    level = int(match1.group(2))
                    gender = match1.group(3)
                    curr_hp = int(match1.group(4))
                    max_hp = int(match1.group(5))

                elif match3:
                    pokemon_name = match3.group(1)
                    level = int(match3.group(2))
                    curr_hp = int(match3.group(3))
                    max_hp = int(match3.group(4))
                    gender = None
                
                # Covers a case where the other two regexes don't do, which is more brute-force. Could consider just defaulting to this
                else:
                    line = p1_info.split('|')
                    player_number = int(line[2][1])
                    pokemon_name = line[2].split(' ')[1]
                    level = int(line[3].split(' ')[1][1:])
                    hp_numerator = int(line[4].split('/')[0])
                    hp_denominator = int(line[4].split('/')[1])
                
                curr_p1_pokemon = Pokemon(gen = 8, species = pokemon_name) #this constructor automatically initializes some info for us

                curr_battle._team[pokemon_name.replace(" ","").lower()] = curr_p1_pokemon
                curr_battle._team[pokemon_name.replace(" ","").lower()]._active = True

                # init some other stuff
                curr_p1_pokemon._level = level
                curr_p1_pokemon._active = True
                curr_p1_pokemon._current_hp = curr_hp
                curr_p1_pokemon._max_hp = max_hp
                curr_p1_pokemon._first_turn = True
                curr_p1_pokemon._status = None
                
                if match2:
                    pokemon_name = match2.group(1)
                    level = int(match2.group(2))
                    gender = match2.group(3)
                    curr_hp = int(match2.group(4))
                    max_hp = int(match2.group(5))
                elif match4:
                    pokemon_name = match4.group(1)
                    level = int(match4.group(2))
                    curr_hp = int(match4.group(3))
                    max_hp = int(match4.group(4))

                # similar to above
                else:
                    line = p2_info.split('|')
                    player_number = int(line[2][1])
                    pokemon_name = line[2].split(' ')[1]
                    level = int(line[3].split(' ')[1][1:])
                    hp_numerator = int(line[4].split('/')[0])
                    hp_denominator = int(line[4].split('/')[1])

                curr_p2_pokemon = Pokemon(gen = 8, species = pokemon_name) #this constructor automatically initializes some info for us
                curr_battle._opponent_team[pokemon_name.replace(" ","").lower()] = curr_p2_pokemon
                curr_battle._opponent_team[pokemon_name.replace(" ","").lower()]._active = True

                # init some other stuff
                curr_p2_pokemon._level = level
                curr_p2_pokemon._current_hp = curr_hp
                curr_p2_pokemon._max_hp = max_hp
                curr_p2_pokemon._first_turn = True
                curr_p2_pokemon._status = None
                
            if line.startswith("|-weather|"):
                weather_regex = r"^\|-weather\|(\w+)\|\[from\]\s(\w+):\s(.*?)\|\[of\]\sp(\d+)a:\s(.*?)$"

                match = re.search(weather_regex, line)
                if match:
                    weather = match.group(1)
                    source_type = match.group(2)
                    source = match.group(3)
                    player = int(match.group(4))
                    pokemon_name = match.group(5)
                    curr_battle._weather = Weather.from_name(weather)
                
                if 'none' in line:
                    curr_battle._weather = None

            # the switch logic is as follows:
            # if the pokemon switching in hasn't been seen yet, initialize a new pokemon and add it to the appropriate team
            if line.startswith("|switch|"):
                switch_regex_pattern = r'\|switch\|p(\d)a: (\w+)\|\w+, L(\d+), \w+\|(\d+)/(\d+)'
                match1 = re.search(switch_regex_pattern, line)

                if match1:
                    player_number = int(match1.group(1))
                    pokemon_name = match1.group(2)
                    level = match1.group(3)
                    hp_numerator = match1.group(4)
                    hp_denominator = match1.group(5)
                else:
                    line = line.split('|')
                    player_number = int(line[2][1])
                    pokemon_name = line[2].split(' ')[1]
                    level = int(re.sub(r'\D', '', line[3].split(' ')[1][1:]))
                    hp_numerator = int(line[4].split('/')[0])
                    hp_denominator = int(line[4].split('/')[1])

                if player_number == 1:
                    if pokemon_name not in curr_battle.team:

                        curr_battle._team[curr_p1_pokemon._species]._active = False

                        curr_p1_pokemon = Pokemon(gen = 8, species = pokemon_name)

                        curr_battle._team[pokemon_name.replace(" ","").lower()] = curr_p1_pokemon
                        curr_battle._team[pokemon_name.replace(" ","").lower()]._active = True

                        curr_p1_pokemon._level = level
                        curr_p1_pokemon._active = True
                        curr_p1_pokemon._current_hp = curr_hp
                        curr_p1_pokemon._max_hp = max_hp
                        curr_p1_pokemon._first_turn = True
                        curr_p1_pokemon._status = None
                    else:
                        curr_battle.active_pokemon = curr_battle.team[pokemon_name]
                else:
                    if pokemon_name not in curr_battle.opponent_team:

                        curr_battle._opponent_team[curr_p2_pokemon._species]._active = False

                        curr_p2_pokemon = Pokemon(gen = 8, species = pokemon_name)
                        curr_battle._opponent_team[pokemon_name.replace(" ","").lower()] = curr_p2_pokemon
                        curr_battle._opponent_team[pokemon_name.replace(" ","").lower()]._active = True

                        curr_p2_pokemon._level = level
                        curr_p2_pokemon._active = True
                        curr_p2_pokemon._current_hp = curr_hp
                        curr_p2_pokemon._max_hp = max_hp
                        curr_p2_pokemon._first_turn = True
                        curr_p2_pokemon._status = None
                    else:
                        pass

            if '|-damage|' in line:
                if 'fnt' not in line:
                    line = line.split('|')
                    player = int(line[2][1])
                    curr_hp = int(line[3].split('/')[0])
                    if player == 1:
                        curr_battle.active_pokemon._current_hp = curr_hp
                    else:
                        curr_battle.opponent_active_pokemon._current_hp = curr_hp
                
            if '|-ability|' in line:
                ability_regex = r"p(\d+)a:\s(.*?)\|.*\|(\w+)\|(\w+)$"
                match = re.search(ability_regex, line)

                if match:
                    player = int(match.group(1))
                    pokemon_name = match.group(2)
                    ability = match.group(3)
                    ability_effect = match.group(4)

                    if player == 1:
                        curr_battle.active_pokemon.ability = ability
                        curr_battle.active_pokemon.ability_effect = ability_effect
                    else:
                        curr_battle.opponent_active_pokemon.ability = ability
                        curr_battle.opponent_active_pokemon.ability_effect = ability_effect

            # if the move isn't registered, then register it
            if '|move|' in line:
                
                line = line.split('|')
                
                player_number = line[2][1]
                move = line[3].replace(' ', '').lower()
                move = re.sub(r'\W+', '', move).lower()

                if player_number == '1':
                    if move not in curr_battle.active_pokemon.moves:
                        curr_battle.active_pokemon.moves[move] = Move(move, gen=8)
                    curr_battle.active_pokemon.moves[move].use()
                else:
                    if move not in curr_battle.opponent_active_pokemon.moves:
                        curr_battle.opponent_active_pokemon.moves[move] = Move(move, gen=8)
                    curr_battle.opponent_active_pokemon.moves[move].use()

            # if heal then update the HP. Might consider the other info (like source and amount healed and such)
            if "|-heal|" in line:
                heal = line.split('|')
                player = int(heal[2][1])
                pokemon_name = heal[2].split(' ')[1]
                curr_hp = int(heal[3].split('/')[0])
                max_hp = int(heal[3].split('/')[1].split(' ')[0])

                if player == 1:
                    curr_battle.active_pokemon._current_hp = curr_hp
                else:
                    curr_battle.opponent_active_pokemon._current_hp = curr_hp

            # update the hazard
            if "|-sidestart|" in line:
                hazard_message = line.split('|')[3]
                player = line.split('|')[2][1]

                condition = SideCondition.from_showdown_message(hazard_message)

                if player == 1:
                    if condition in STACKABLE_CONDITIONS:
                        curr_battle.side_conditions[condition] = min(curr_battle.side_conditions[condition] + 1, STACKABLE_CONDITIONS[condition])
                    else:
                        curr_battle.side_conditions[condition] = 1
                else:
                    if condition in STACKABLE_CONDITIONS:
                        curr_battle.opponent_side_conditions[condition] = min(curr_battle.opponent_side_conditions[condition] + 1, STACKABLE_CONDITIONS[condition])
                    else:
                        curr_battle.opponent_side_conditions[condition] = 1

            # update stats
            if "|-boost|" in line:
                boost_regex = r"p(\d+)a:\s\w+\|(\w+)\|(\d+)"

                match = re.search(boost_regex, line)
                player = int(match.group(1))
                stat = match.group(2)
                amount = int(match.group(3))

                if player == 1:
                    curr_battle.active_pokemon.boosts[stat] += amount
                else:
                    curr_battle.opponent_active_pokemon.boosts[stat] += amount

            # update stats
            if "|-unboost|" in line:
                boost_regex = r"p(\d+)a:\s\w+\|(\w+)\|(\d+)"

                match = re.search(boost_regex, line)
                player = int(match.group(1))
                stat = match.group(2)
                amount = int(match.group(3))

                if player == 1:
                    curr_battle.active_pokemon.boosts[stat] -= amount
                else:
                    curr_battle.opponent_active_pokemon.boosts[stat] -= amount

            # faint case, update status
            if "|faint|" in line:
                faint_regex = r"p(\d+)a:\s(.*?)$"

                match = re.search(faint_regex, line)
                player = int(match.group(1))
                pokemon_name = match.group(2)

                if player == 1:
                    curr_battle.active_pokemon._status = Status.FNT
                    curr_battle.active_pokemon._active = False
                    
                else:
                    curr_battle.opponent_active_pokemon._status = Status.FNT
                    curr_battle.opponent_active_pokemon._active = False

            # update status condition        
            if "|-status|" in line:
                status_regex = r"p(\d+)a:\s(.*?)\|(\w+)"

                match = re.search(status_regex, line)
                player = int(match.group(1))
                pokemon_name = match.group(2)
                status = match.group(3)

                if status == 'par':
                    status = Status.PAR
                elif status == 'brn':
                    status = Status.BRN
                elif status == 'psn':
                    status = Status.PSN
                elif status == 'tox':
                    status = Status.TOX
                elif status == 'slp':
                    status = Status.SLP
                elif status == 'frz':
                    status = Status.FRZ
                
                if player == 1:
                    curr_battle.active_pokemon._status = status
                else:
                    curr_battle.opponent_active_pokemon._status = status

            if "|-sideend|" in line:
                hazard_message = line.split('|')[3]
                player = line.split('|')[2][1]

                if player == 1:
                    curr_battle.side_conditions[SideCondition.from_showdown_message(hazard_message)] = 0
                else:
                    curr_battle.opponent_side_conditions[SideCondition.from_showdown_message(hazard_message)] = 0

            if '|turn|' in line:
                battle_log.append(copy.deepcopy(curr_battle))
        
        return battle_log

