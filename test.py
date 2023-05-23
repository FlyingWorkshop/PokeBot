from alphagogoat.delphox import Delphox
from alphagogoat.utils import DataExtractor
from alphagogoat.extractor import process_battle
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import json
import numpy as np

def main():
    battle_json = "cache/replays/gen8randombattle-1123651831.json"
    turns = process_battle(battle_json)


if __name__ == "__main__":
    main()