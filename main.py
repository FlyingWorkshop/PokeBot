import json
import logging
import random
from copy import deepcopy
from pathlib import Path

import torch
from joblib import Parallel, delayed
from poke_env.environment.battle import Battle
from tqdm.auto import tqdm

from alphagogoat.catalogs import MoveEnum
from alphagogoat.constants import LSTM_INPUT_SIZE, DEVICE
from alphagogoat.delphox import Delphox, train, evaluate
from alphagogoat.utils import move_to_pred_vec_index

LOGGER = logging.getLogger('poke-env')

def process_input_log(log):
    """
    >>> log = Path("cache/replays/gen8randombattle-1872565566.log").read_text()
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
                out_me[move_to_pred_vec_index(curr_line[2])] = 1
            elif curr_line[1] == 'switch':
                out_me[-1] = 1

            if next_line[1] == 'move':
                out_them[move_to_pred_vec_index(next_line[2])] = 1
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
    mon1, mon2 = None, None
    for line in history:
        try:
            b._parse_message(line.split('|'))
            # print(b.turn)
            # print(b.active_pokemon)
            # print(b.opponent_active_pokemon)
            # if mon1 is None:
            #     mon1 = b.active_pokemon
            # if mon2 is None:
            #     mon2 = b.opponent_active_pokemon
            if line.split('|')[1] == 'turn':
                # monkey patch issue where fainted pokemon are immediately replaced the same turn
                # b.active_mon = mon1
                # b.opponent_active_mon = mon2
                battles.append(deepcopy(b))
                # mon1, mon2 = None, None
        except:
            continue
    move1, move2 = process_input_log(replay)
    return battles, move1, move2

def main():
    """
    TODO: ideas
    - include ELO in embedding (the model should have different predictions for skilled and unskilled players)
    - less penalty for guessing the wrong move but correct type, category (TODO: make this a damage multiplier difference instead)
    - heavier penalty for guessing switching incorrectly
    - make delphox deeper
    """
    delphox_path = "delphox_test_smallish.pth"
    json_files = [filepath for filepath in Path("cache/replays").iterdir() if filepath.name.endswith('.json')]
    train_files, test_files = json_files[:-10], json_files[-10:]
    reps = 1000
    delphox = Delphox(3029).to(device=DEVICE)
    for _ in range(reps):
        random.shuffle(train_files)
        #print(len(train_files))
        # train_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(train_files))
        # train_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(train_files[:100]))  # MEDIUM
        train_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(train_files[:30]))  # SMALL
        # train_data = [make_data(f) for f in tqdm(json_files[:1])]  # SINGLE-PROCESS DEBUGGING
        #train_data = [make_data("cache/replays/gen8randombattle-1875194808.json")]
        # if Path(delphox_path).exists():
        #     delphox.load_state_dict(torch.load(delphox_path))
        #     delphox.eval()
        train(delphox, train_data, lr=0.1, discount=0, weight_decay=0.001)
        #torch.save(delphox.state_dict(), delphox_path)
    test_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(test_files))
    evaluate(delphox, test_data)



if __name__ == "__main__":
    main()

