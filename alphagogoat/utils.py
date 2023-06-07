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


def vec2str(vec: torch.Tensor, pokemon: Pokemon, team: dict[str, Pokemon]):
    i = vec.argmax().item()
    if i >= MAX_MOVES:
        i -= MAX_MOVES
        team = sorted([mon.species for mon in team.values()])
        if i >= len(team):
            return "unseen"
        else:
            return team[i]
    else:
        moves = sorted(POKEDEX[pokemon.species]["moves"])
        if i > len(moves):
            print(i, vec, pokemon, moves)
        return moves[i]


def process_line(line: str):
    if "switch" in line:
        pokemon_switch = re.search(r"p[12]a: (.*?)\|", line).groups(0)[0]
        pokemon_switch = re.sub(r"[-’\s\.:]", "", pokemon_switch.lower())
        return ("switch", pokemon_switch)
    else:
        pokemon_move = re.search(r"\|([A-Z].*?)\|", line).groups(0)[0]
        pokemon_move = re.sub(r"\s|-|'", "", pokemon_move.lower())
        return ("move", pokemon_move)

def get_actions(filepath: str):
    actions1, actions2 = [], []
    with open(filepath) as f:
        data = json.load(f)
    turn_texts = data['log'].split('|turn|')[1:]
    for text in turn_texts:
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
    return actions1, actions2

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
    vectors1 = []
    vectors2 = []
    for turn, a1, a2 in zip(battles, actions1, actions2):
        # TODO: handle ditto
        if turn.active_pokemon.species == 'ditto' or turn.opponent_active_pokemon.species == 'ditto':
            continue

        # skip dynamx TODO: handle dynamax
        if a1[0] == 'move' and a1[1] not in POKEDEX[turn.active_pokemon.species]["moves"]:
            continue
        if a2[0] == 'move' and a2[1] not in POKEDEX[turn.opponent_active_pokemon.species]["moves"]:
            continue

        v1 = torch.zeros(MAX_MOVES + NUM_POKEMON_PER_TEAM)
        if a1[0] == 'switch':
            team = sorted([mon.species for mon in turn.team.values()])
            if a1[1] in team:
                v1[MAX_MOVES + team.index(a1[1])] = 1
            else:
                # switch to an unseen pokemon
                v1[MAX_MOVES + len(team) - 1] = 1
            vectors1.append(v1)
        else:
            moves = sorted(POKEDEX[turn.active_pokemon.species]["moves"])
            v1[moves.index(a1[1])] = 1
            vectors1.append(v1)

        v2 = torch.zeros(MAX_MOVES + NUM_POKEMON_PER_TEAM)
        if a2[0] == 'switch':
            team = sorted([mon.species for mon in turn.opponent_team.values()])
            if a2[1] in team:
                v2[MAX_MOVES + team.index(a2[1])] = 1
            else:
                # switch to an unseen pokemon
                v2[MAX_MOVES + len(team) - 1] = 1
            vectors2.append(v2)
        else:
            moves = sorted(POKEDEX[turn.opponent_active_pokemon.species]["moves"])
            v2[moves.index(a2[1])] = 1
            vectors2.append(v2)

        turns.append(turn)

    return turns, vectors1, vectors2


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

