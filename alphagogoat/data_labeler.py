from utils import DataExtractor
import json
from tqdm import tqdm
import pathlib
from pathlib import Path
import multiprocessing as mp
from catalogs import *
import torch


def process_input_log(log):
    input_log = log['inputlog'].split('\n')

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
        if curr_line.startswith('>p1') and next_line.startswith('>p2'):  # that means this is a normal turn with no fainting or anything
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

def create_labeled_data():
    """
    Goes through all the json's present in the cache/replays folder, creates a dictionary mapping from a Battle object representing each turn to the pair of actions that was taken (move/switch)
    The tuple is represented as (action1, action2) where action1 is the action taken by the player whose team is listed first in the replay file, ignoring which player actually went first
    If a status condition prevents a move (like FRZ, SLP, FNT), then the action is None
    """
    
    data = {}

    for filepath in tqdm(list(Path("/Users/adamzhao/Desktop/PokeBot/cache/replays").iterdir())):
        if str(filepath).endswith('.json'):
            battle = DataExtractor(str(filepath)).turns
            with open(filepath, 'r') as f:
                replay = json.load(f)
                turn_actions = process_input_log(replay)

            k, v = [],[]
            for turn, action in zip(battle, turn_actions):
                #data[turn] = action
                k.append(turn)
                v.append(action)
            
            data[tuple(k)] = tuple(v)

    return data


DATASET = create_labeled_data()
        

        
            
