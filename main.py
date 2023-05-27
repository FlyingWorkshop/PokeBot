from alphagogoat.embedder import process_battle, get_team_histories
from alphagogoat.catalogs import MoveEnum
from alphagogoat.delphox import Delphox, train
from alphagogoat.constants import LSTM_INPUT_SIZE, LSTM_OUTPUT_SIZE, device


from poke_env.environment.battle import Battle
from tqdm.auto import tqdm
import torch

import json
import logging
from copy import deepcopy
from pathlib import Path
from joblib import Parallel, delayed
import re


LOGGER = logging.getLogger('poke-env')


def process_input_log(log):
    """
    >>> log = Path("cache/replays/gen8randombattle-1123651831.log").read_text()
    >>> process_input_log(log)
    """
    input_log = log['inputlog']
    input_log = input_log.split('\n')
    start = 0
    for line in input_log:
        if line.startswith('>p1'):
            break
        start += 1

    input_log = input_log[start:]
    out1 = []
    out2 = []

    for i in range(len(input_log) - 1):
        curr_line = input_log[i]
        next_line = input_log[i+1]
        if curr_line.startswith('>p1') and next_line.startswith('>p2'): # that means this is a normal turn with no fainting or anything
            out_me = torch.zeros(len(MoveEnum) + 1)
            out_them = torch.zeros(len(MoveEnum) + 1)
            curr_line = curr_line.split(' ')
            next_line = next_line.split(' ')

            if curr_line[1] == 'move':
                out_me[MoveEnum[re.sub(r"\s|-|'", "", curr_line[2].lower())].value - 1] = 1
            elif curr_line[1] == 'switch':
                out_me[-1] = 1

            if next_line[1] == 'move':
                #print(re.sub(r"\s|-|'", "", next_line[2].lower()))
                out_them[MoveEnum[re.sub(r"\s|-|'", "", next_line[2].lower())].value - 1] = 1
            elif next_line[1] == 'switch':
                out_them[-1] = 1
            i += 1
            out1.append(out_me)
            out2.append(out_them)
        else:
            continue

    return out1, out2

def make_data(filepath):
    with open(filepath) as f:
        replay = json.load(f)
    history = replay['log'].split('\n')
    battles = []
    b = Battle(replay['id'], replay['p1'], LOGGER, 8)
    b._opponent_username = replay['p2']
    for line in history:
        try:
            b._parse_message(line.split('|'))
            if line.split('|')[1] == 'turn':
                battles.append(deepcopy(b))
        except:
            continue
    move1, move2 = process_input_log(replay)
    h1, h2 = get_team_histories(battles)
    return battles, h1, h2, move1, move2


def main():
    json_files = [filepath for filepath in Path("cache/replays").iterdir() if filepath.name.endswith('.json')]
    dataset = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(json_files))
    delphox = Delphox(LSTM_INPUT_SIZE).to(device=device)
    train(delphox, dataset)


if __name__ == "__main__":
    main()

