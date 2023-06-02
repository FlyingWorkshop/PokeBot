import json
import logging
import re
from copy import deepcopy

import torch
from poke_env.environment.battle import Battle

from .catalogs import MoveEnum

LOGGER = logging.getLogger('poke-env')

def move_to_pred_vec_index(m):
    return MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1


def vec2str(pred: torch.Tensor):
    i = pred.argmax().item()
    if i == len(MoveEnum):
        return 'switch'
    else:
        return MoveEnum(i + 1).name



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

def make_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    history = data['log'].split('\n')
    battles = []
    b = Battle(data['id'], data['p1'], LOGGER, 8)
    b._opponent_username = data['p2']
    # mon1, mon2 = None, None
    for line in history:
        try:
            b._parse_message(line.split('|'))
            # if mon1 is None:
            #     mon1 = b.active_pokemon
            # if mon2 is None:
            #     mon2 = b.opponent_active_pokemon
            if line.split('|')[1] == 'turn':
                # monkey patch issue where fainted Pokémon are immediately replaced the same turn
                # b.active_mon = mon1
                # b.opponent_active_mon = mon2
                battles.append(deepcopy(b))
                # mon1, mon2 = None, None
        except:
            continue

    actions1, actions2 = [], []
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
    return battles, actions1, actions2