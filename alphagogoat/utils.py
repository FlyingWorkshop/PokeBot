import re
from .catalogs import MoveEnum
import json

import torch

def move_to_pred_vec_index(m):
    return MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1


def pred_vec_to_string(pred: torch.Tensor):
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