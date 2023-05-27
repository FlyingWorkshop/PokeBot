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

def main():
    json_files = [filepath for filepath in Path("cache/replays").iterdir() if filepath.name.endswith('.json')]
    dataset = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(json_files))
    delphox = Delphox(LSTM_INPUT_SIZE).to(device=device)
    train(delphox, dataset)


if __name__ == "__main__":
    main()

