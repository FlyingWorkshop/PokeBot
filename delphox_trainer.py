import json
import logging
import random
from pathlib import Path

import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from alphagogoat.delphox import Delphox, train
from alphagogoat.utils import make_delphox_data


def main():
    delphox_path = "delphox.pth"
    json_files = [filepath for filepath in Path("cache/replays").iterdir() if filepath.name.endswith('.json')]
    train_files, test_files = json_files[:-100], json_files[-100:]
    reps = 100
    delphox = Delphox(2825)
    # random.shuffle(train_files)
    # train_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(train_files))
    # train_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(train_files[:100]))  # MEDIUM
    # train_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(train_files[:30]))  # SMALL
    # train_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(train_files[:8]))  # MINI
    # train_data = [make_delphox_data(f) for f in tqdm(["cache/replays/gen8randombattle-1878695089.json"])]  # SINGLE-PROCESS DEBUGGING
    # if Path(delphox_path).exists():
    #     delphox.load_state_dict(torch.load(delphox_path))
    for _ in range(reps):
        random.shuffle(train_files)
        # train_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(train_files[:100]))
        train_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(train_files[:8]))
        train(delphox, train_data, lr=0.001, weight_decay=0, discount=0)
        # torch.save(delphox.state_dict(), delphox_path)
    # test_data = Parallel(n_jobs=4)(delayed(make_delphox_data)(filepath) for filepath in tqdm(test_files))
    # evaluate(victini, test_data)


if __name__ == "__main__":
    main()

