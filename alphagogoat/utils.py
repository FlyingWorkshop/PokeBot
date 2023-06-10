import json
import logging
import re
from copy import deepcopy

import torch
from poke_env.environment.battle import Battle, Pokemon

from .catalogs import MoveEnum
from .constants import MAX_MOVES, NUM_POKEMON_PER_TEAM
from .pokedex import POKEDEX

LOGGER = logging.getLogger('poke-env')

def vec2action(vec: torch.Tensor, turn: Battle, opponent_pov: bool):
    if opponent_pov:
        pokemon = turn.opponent_active_pokemon
        team = turn.opponent_team
    else:
        pokemon = turn.active_pokemon
        team = turn.team

    i = vec.argmax().item()
    if i >= MAX_MOVES:
        i -= MAX_MOVES
        team = sorted([mon.species for mon in team.values()])
        if i == len(team):
            return ("switch", "unseen")
        else:
            return ("switch", team[i])
    else:
        moves = sorted(POKEDEX[pokemon.species]["moves"])
        return ("move", moves[i])


def process_line(line: str):
    if "switch" in line:
        pokemon_switch = re.search(r"p[12]a: (.*?)\|", line).groups(0)[0]
        pokemon_switch = re.sub(r"[-’\s\.:]", "", pokemon_switch.lower())
        return ("switch", pokemon_switch)
    else:
        pokemon_move = re.search(r"\|([A-Z].*?)\|", line).groups(0)[0]
        pokemon_move = re.sub(r"\s|-|'", "", pokemon_move.lower())
        return ("move", pokemon_move)


def action2vec(action: tuple[str, str], team: dict[str: Pokemon], active: Pokemon):
    vec = torch.zeros(MAX_MOVES + NUM_POKEMON_PER_TEAM)
    if action[0] == 'move':
        # TODO: handle dynamax and transform
        # TODO: handle moves like u-turn
        moves = sorted(POKEDEX[active.species]['moves'])
        selected_move = action[1]
        i = moves.index(selected_move)
        vec[i] = 1
    else:  # action[0] == 'switch':
        team = sorted([mon.species for mon in team.values()])
        switched_in = action[1]
        for j, species in enumerate(team):
            # handles problem where the log records pokemon like 'moltres-galar' as 'moltres'
            if species.startswith(switched_in):
                i = j
                break
        else:
            # unseen pokemon
            i = MAX_MOVES + len(team)
        vec[i] = 1
    return vec

def make_delphox_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    history = data['log'].split('\n')
    battles = []
    b = Battle(data['id'], data['p1'], LOGGER, 8)
    b._opponent_username = data['p2']
    for line in history:
        try:
            b._parse_message(line.split('|'))
            if line.split('|')[1] == 'turn':
                battles.append(deepcopy(b))
        except:
            continue

    actions1, actions2 = [], []
    turn_texts = data['log'].split('|turn|')[1:]
    for i, text in enumerate(turn_texts):
        matches = re.findall(r"(\|[ms].+\|)", text)
        for m in matches:
            if "|move|p1a:" in m or "|switch|p1a:" in m:
                cooked = process_line(m)
                actions1.append(cooked)
                break
        for m in matches:
            if "|move|p2a:" in m or "|switch|p2a:" in m:
                cooked = process_line(m)
                actions2.append(cooked)
                break

    # TODO: change later maybe?
    turns = []
    clean_actions1 = []
    clean_actions2 = []
    for turn, a1, a2 in zip(battles, actions1, actions2):
        if turn.active_pokemon.species == 'ditto' or turn.opponent_active_pokemon.species == 'ditto':
            continue

        # skip dynamx TODO: handle dynamax
        if a1[0] == 'move' and a1[1] not in POKEDEX[turn.active_pokemon.species]["moves"]:
            continue
        if a2[0] == 'move' and a2[1] not in POKEDEX[turn.opponent_active_pokemon.species]["moves"]:
            continue

        # if a1[0] == 'switch' or a2[0] == 'switch':
        #     continue

        turns.append(turn)
        clean_actions1.append(a1)
        clean_actions2.append(a2)



    # return turns[-10:], clean_actions1[-10:], clean_actions2[-10:]
    return turns, clean_actions1, clean_actions2



def make_victini_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    history = data['log'].split('\n')
    battles = []
    for line in history:
        if "|win|" in line:
            won = data['p1'] in line
            break
    else:
        print(f"Could not find winner in {filepath}.")
    b = Battle(data['id'], data['p1'], LOGGER, 8)
    b._opponent_username = data['p2']

    for line in history:
        try:
            b._parse_message(line.split('|'))
            if line.split('|')[1] == 'turn':
                battles.append(deepcopy(b))
        except:
            continue
    return battles[-10:], won
