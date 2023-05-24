from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from enum import IntEnum

def _make_pokedex():
    pokedex = {}
    for filepath in tqdm(list(Path("./../cache/teams").iterdir())):
        with open(filepath, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if key not in pokedex:
                pokedex[key] = {k: [] for k in value.keys()}
            for k, v in value.items():
                pokedex[key][k].append(v)

    for pokemon, dicts in tqdm(pokedex.items()):
        pokedex[pokemon]['level'] = int(np.round(np.mean(pokedex[pokemon]['level'])))
        for k1 in ['abilities', 'moves', 'items']:
            d = {}
            if k1 in dicts:
                for dict_ in dicts[k1]:
                    if not isinstance(dict_, dict):
                        continue
                    for k2, prob in dict_.items():
                        if k2 not in d:
                            d[k2] = []
                        d[k2].append(prob)
                for k2, probs in d.items():
                    d[k2] = np.mean(probs)
            pokedex[pokemon][k1] = d

    return pokedex


POKEDEX = _make_pokedex()