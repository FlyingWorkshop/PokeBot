from utils import DataExtractor
import json
from tqdm import tqdm
import pathlib
from pathlib import Path
import multiprocessing as mp


def process_input_log(log) -> list[tuple]:
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
            pair = []
            curr_line = curr_line.split(' ')
            next_line = next_line.split(' ')

            if curr_line[1] == 'move':
                pair.append(curr_line[2])
            elif curr_line[1] == 'switch':
                pair.append(curr_line[1])
            else:
                pair.append(None)
            
            if next_line[1] == 'move':
                pair.append(next_line[2])
            elif next_line[1] == 'switch':
                pair.append(next_line[1])
            else:
                pair.append(None)
            
            i += 1
            out.append(tuple(pair))
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

    for filepath in tqdm(list(Path("../cache/replays").iterdir())):
        if str(filepath).endswith('.json'):
            battle = DataExtractor(str(filepath)).turns
            with open(filepath, 'r') as f:
                replay = json.load(f)
                turn_actions = process_input_log(replay)
            
            for turn, action in zip(battle, turn_actions):
                data[turn] = action

    return data


DATASET = create_labeled_data()
        

        
            
