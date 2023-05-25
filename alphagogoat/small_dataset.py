from tqdm import tqdm
from pathlib import Path
import json
from utils import DataExtractor
from catalogs import *
import torch


def process_input_log(log):
    input_log = log['inputlog']

    input_log = input_log.split('\n')

    start = 0

    for line in input_log:
        if line.startswith('>p1'):
            break
        start += 1
    
    input_log = input_log[start:]
    out = []

    for i in range(len(input_log) - 1):
        curr_line = input_log[i]
        next_line = input_log[i+1]
        if curr_line.startswith('>p1') and next_line.startswith('>p2'): # that means this is a normal turn with no fainting or anything
            out_me = torch.zeros(len(MoveEnum) + 1)
            out_them = torch.zeros(len(MoveEnum) + 1)
            curr_line = curr_line.split(' ')
            next_line = next_line.split(' ')

            if curr_line[1] == 'move':
                out_me[MoveEnum[curr_line[2].lower()].value - 1] = 1
            elif curr_line[1] == 'switch':
                out_me[-1] = 1
            
            if next_line[1] == 'move':
                out_them[MoveEnum[next_line[2].lower()].value - 1] = 1
            elif next_line[1] == 'switch':
                out_them[-1] = 1
    

            i += 1
            out.append(torch.cat((out_me, out_them)))
        else:
            continue

    return out

SMALL_DATASET = {}
i = 0
for filepath in tqdm(list(Path("./../cache/replays").iterdir())):
    if i == 1:
        break
    if str(filepath).endswith('.json'):
        i += 1
        battle = DataExtractor(str(filepath)).turns
        with open(filepath, 'r') as f:
            replay = json.load(f)
            turn_actions = process_input_log(replay)

        k, v = [],[]
        for turn, action in zip(battle, turn_actions):
            #data[turn] = action
            k.append(turn)
            v.append(action)
        
        SMALL_DATASET[tuple(k)] = tuple(v)