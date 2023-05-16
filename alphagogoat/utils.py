import json
import re
from poke_env.environment.abstract_battle import AbstractBattle
import pypokedex
import logging
from poke_env.environment.pokemon import Pokemon, PokemonType
import pypokedex
from poke_env.environment.weather import Weather
from poke_env.environment.move import Move
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
import copy
from poke_env.player_configuration import PlayerConfiguration
from poke_env.environment.battle import Battle



def _get_turns(log) -> list:
    turns = []
    turn = []
    for line in log.split('\n'):
        if line.startswith("|turn|") and turn:
            turns.append(turn)
            turn = []
        turn.append(line)
    return turns


def battle_process(battle_log: str) -> list[Battle]:
    logger = logging.getLogger('poke-env')
    with open(battle_log) as f:
        battle_data = json.load(f)
    
    history = battle_data['log']

    history = history.split('\n')

    curr_p1_pokemon = None
    curr_p2_pokemon = None

    curr_battle = Battle(battle_data['id'], battle_data['p1'],logger, gen)

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
        
        if line.startswith("|start"): # start of battle
            #print(line)
            switch_regex_pattern1 = r"p1a:\s(.*?)\|.*?L(\d+).*?(M|F)\|(\d+)/(\d+)"
            switch_regex_pattern2 = r"p2a:\s(.*?)\|.*?L(\d+).*?(M|F)\|(\d+)/(\d+)"
            p1_info = history[idx + 1]
            p2_info = history[idx + 2]
            # print(p1_info)
            # print(p2_info)
            to_skip.append(idx + 1)
            to_skip.append(idx + 2)
            match1 = re.search(switch_regex_pattern1, p1_info)
            match2 = re.search(switch_regex_pattern2, p2_info)
            
            if match1:
                #print("match1")
                pokemon_name = match1.group(1)
                level = int(match1.group(2))
                gender = match1.group(3)
                curr_hp = int(match1.group(4))
                max_hp = int(match1.group(5))

                pokemon1_info = pypokedex.get(name=pokemon_name)

                #generation_dict.get(pokemon2_info.dex, 9)
                curr_p1_pokemon = Pokemon(gen = 8, species = pokemon1_info.name)

                curr_battle.team[pokemon1_info.name] = curr_p1_pokemon

                curr_p1_pokemon._level = level
                curr_p1_pokemon._possible_abilities = [p.name for p in pokemon1_info.abilities]
                curr_p1_pokemon._heightm = pokemon1_info.height
                curr_p1_pokemon._weightkg = pokemon1_info.weight
                curr_p1_pokemon._type_1 = PokemonType.from_name(pokemon1_info.types[0])
                if len(pokemon1_info.types) > 1:
                    curr_p1_pokemon._type_2 = PokemonType.from_name(pokemon1_info.types[1])
                curr_p1_pokemon._active = True
                curr_p1_pokemon._current_hp = curr_hp
                curr_p1_pokemon._max_hp = max_hp
                curr_p1_pokemon._first_turn = True
                curr_p1_pokemon._status = None

                curr_p1_pokemon._update_from_pokedex(species=pokemon1_info.name)
                #print(pokemon_name, level, curr_hp, max_hp)
            else:
                print("no match")
            
            if match2:
                #print("match2")
                pokemon_name = match2.group(1)
                level = int(match2.group(2))
                gender = match2.group(3)
                curr_hp = int(match2.group(4))
                max_hp = int(match2.group(5))

                pokemon2_info = pypokedex.get(name=pokemon_name)

                #generation_dict.get(pokemon2_info.dex, 9)
                curr_p2_pokemon = Pokemon(gen = 8, species = pokemon2_info.name)
                curr_battle.opponent_team[pokemon2_info.name] = curr_p2_pokemon

                curr_p2_pokemon._level = level
                curr_p2_pokemon._possible_abilities = [p.name for p in pokemon2_info.abilities]
                curr_p2_pokemon._heightm = pokemon2_info.height
                curr_p2_pokemon._weightkg = pokemon2_info.weight
                curr_p2_pokemon._type_1 = PokemonType.from_name(pokemon2_info.types[0])
                if len(pokemon2_info.types) > 1:
                    curr_p2_pokemon._type_2 = PokemonType.from_name(pokemon2_info.types[1])
                curr_p2_pokemon._active = True
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
                curr_battle.weather = Weather.from_name(weather)
            
            if 'none' in line:
                curr_battle.weather = None


        if line.startswith("|switch|"):
            switch_regex_pattern = r'\|switch\|p(\d)a: (\w+)\|\w+, L(\d+), \w+\|(\d+)/(\d+)'
            match1 = re.search(switch_regex_pattern, line)

            if match1:
                player_number = match.group(1)
                pokemon_name = match.group(2)
                # level = match.group(3)
                hp_numerator = match.group(3)
                hp_denominator = match.group(4)

                if player_number == 1:
                    curr_battle.active_pokemon._active = False
                    if pokemon_name not in curr_battle.team:

                        pokemon1_info = pypokedex.get(name=pokemon_name)

                        curr_p1_pokemon = Pokemon(gen = 8, species = pokemon1_info.name)

                        curr_battle.team[pokemon1_info.name] = curr_p1_pokemon

                        curr_battle.active_pokemon = curr_p1_pokemon

                        curr_p1_pokemon.level = level
                        curr_p1_pokemon._possible_abilities = [p.name for p in pokemon1_info.abilities]
                        curr_p1_pokemon._heightm = pokemon1_info.height
                        curr_p1_pokemon._weightkg = pokemon1_info.weight
                        curr_p1_pokemon._type_1 = PokemonType.from_name(pokemon1_info.types[0].name)
                        if len(pokemon1_info.types) > 1:
                            curr_p1_pokemon._type_2 = PokemonType.from_name(pokemon1_info.types[1].name)
                        curr_p1_pokemon._active = True
                        curr_p1_pokemon._current_hp = curr_hp
                        curr_p1_pokemon._max_hp = max_hp
                        curr_p1_pokemon._first_turn = True
                        curr_p1_pokemon._status = None

                        curr_p1_pokemon._update_from_pokedex(species=pokemon1_info.name)
                        #print(pokemon_name, level, curr_hp, max_hp)
                    else:
                        curr_battle.active_pokemon = curr_battle.team[pokemon_name]
                else:
                    curr_battle.opponent_active_pokemon._active = False
                    if pokemon_name not in curr_battle.opponent_team:
                        pokemon2_info = pypokedex.get(name=pokemon_name)

                        curr_p2_pokemon = Pokemon(gen = 8, species = pokemon2_info.name)
                        curr_battle.opponent_team[pokemon2_info.name] = curr_p2_pokemon

                        curr_battle.opponent_active_pokemon = curr_p2_pokemon
                        curr_battle.opponent_active_pokemon._active = True
                        curr_p2_pokemon.level = level
                        curr_p2_pokemon._possible_abilities = [p.name for p in pokemon2_info.abilities]
                        curr_p2_pokemon._heightm = pokemon2_info.height
                        curr_p2_pokemon._weightkg = pokemon2_info.weight
                        curr_p2_pokemon._type_1 = PokemonType.from_name(pokemon2_info.types[0].name)
                        if len(pokemon2_info.types) > 1:
                            curr_p2_pokemon._type_2 = PokemonType.from_name(pokemon2_info.types[1].name)
                        curr_p2_pokemon._active = True
                        curr_p2_pokemon._current_hp = curr_hp
                        curr_p2_pokemon._max_hp = max_hp
                        curr_p2_pokemon._first_turn = True
                        curr_p2_pokemon._status = None

                        curr_p2_pokemon._update_from_pokedex(species=pokemon2_info.name)
                    
                    else:
                        curr_battle.active_pokemon = curr_battle.opponent_team[pokemon_name]
            else:
                print("no match")


        if line.startswith("|-damage|"):
            
            damage_regex = r"p(\d+)a:\s(.*?)\|(\d+)/(\d+)$"

            if 'fnt' not in line:
                match = re.search(damage_regex, line)
                player = int(match.group(1))
                pokemon_name = match.group(2)
                curr_hp = int(match.group(3))
                max_hp = int(match.group(4))

                if player == 1:
                    curr_battle.active_pokemon.current_hp = curr_hp
                else:
                    curr_battle.opponent_active_pokemon.current_hp = curr_hp
            

        if line.startswith("|-ability|"):
            ability_regex = r"p(\d+)a:\s(.*?)\|.*\|(\w+)\|(\w+)$"

            match = re.search(ability_regex, line)
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

        if line.startswith("|move|"):
            
            line = line.split('|')
            
            player_number = line[2][1]
            move = line[3].replace(' ', '').lower()

            if player_number == '1':
                if move not in curr_battle.active_pokemon.moves:
                    curr_battle.active_pokemon.moves[move] = Move(move, gen=8)
                curr_battle.active_pokemon.moves[move].use()
            else:
                #if move not in curr_battle.opponent_active_pokemon.moves:
                curr_battle.opponent_active_pokemon.moves[move] = Move(move, gen=8)
                curr_battle.opponent_active_pokemon.moves[move].use()

        if line.startswith("|-heal|"):
            heal_regex = r"p(\d+)a:\s(.*?)\|(\d+)/(\d+)\|\[from\]\s(\w+):\s(\w+)"

            match = re.search(heal_regex, line)
            player = int(match.group(1))
            pokemon_name = match.group(2)
            curr_hp = int(match.group(3))
            max_hp = int(match.group(4))
            source_type = match.group(5)
            source = match.group(6)

            if player == 1:
                curr_battle.active_pokemon.current_hp = curr_hp
            else:
                curr_battle.opponent_active_pokemon.current_hp = curr_hp

        if line.startswith("|-sidestart|"): # hazards
            hazard_regex = r"p(\d+):.*?move:\s(.*?)$"

            match = re.search(hazard_regex, line)
            player = int(match.group(1))
            hazard = match.group(2)

            if player == 1:
                curr_battle.side_conditions[SideCondition.from_name(hazard)] = True
            else:
                curr_battle.opponent_side_conditions[SideCondition.from_name(hazard)] = True

        if line.startswith("|-boost|"):
            boost_regex = r"p(\d+)a:\s\w+\|(\w+)\|(\d+)"

            match = re.search(boost_regex, line)
            player = int(match.group(1))
            stat = match.group(2)
            amount = int(match.group(3))

            if player == 1:
                curr_battle.active_pokemon.boosts[stat] += amount
            else:
                curr_battle.opponent_active_pokemon.boosts[stat] += amount

        if line.startswith("|-unboost|"):
            boost_regex = r"p(\d+)a:\s\w+\|(\w+)\|(\d+)"

            match = re.search(boost_regex, line)
            player = int(match.group(1))
            stat = match.group(2)
            amount = int(match.group(3))

            if player == 1:
                curr_battle.active_pokemon.boosts[stat] -= amount
            else:
                curr_battle.opponent_active_pokemon.boosts[stat] -= amount


        if line.startswith("|faint|"):
            faint_regex = r"p(\d+)a:\s(.*?)$"

            match = re.search(faint_regex, line)
            player = int(match.group(1))
            pokemon_name = match.group(2)

            if player == 1:
                curr_battle.active_pokemon._active = False
                curr_battle.active_pokemon.status = Status.FNT
            else:
                curr_battle.opponent_active_pokemon._active = False
                curr_battle.opponent_active_pokemon.status = Status.FNT

        if line.startswith("|-status|"):
            status_regex = r"p(\d+)a:\s(.*?)\|(\w+)"

            match = re.search(status_regex, line)
            player = int(match.group(1))
            pokemon_name = match.group(2)
            status = match.group(3)

            if player == 1:
                curr_battle.active_pokemon.status = Status.from_name(status)
            else:
                curr_battle.opponent_active_pokemon.status = Status.from_name(status)

        if line.startswith("|win|"):
            pass

        battle_log.append(copy.deepcopy(curr_battle))
    
    return battle_log

