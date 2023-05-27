from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from enum import IntEnum

import re

def _make_pokedex():
    pokedex = {}
    folder = Path(__file__).parent.parent / "cache" / "teams"
    for filepath in tqdm(list(folder.iterdir())):
        with open(filepath, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if key not in pokedex:
                key = re.sub(r"[-’\s\.]", "", key.lower())
                pokedex[key] = {k: [] for k in value.keys()}
            for k, v in value.items():
                pokedex[key][k].append(v)

    for species, dicts in tqdm(pokedex.items()):
        pokedex[species]['level'] = int(np.round(np.mean(pokedex[species]['level'])))
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
            pokedex[species][k1] = d

    for species, data in pokedex.items():
        if 'evs' in data:
            pokedex[species]['evs'] = data['evs'][0]
        if 'ivs' in data:
            del pokedex[species]['ivs']

    mons = list(pokedex.keys())
    for species in mons:
        if 'gmax' in species:
            pokedex[species[:-4]] = pokedex[species]
            del pokedex[species]

    pokedex['zygarde10'] = pokedex['zygarde10%']
    del pokedex['zygarde10%']

    pokedex['gastrodoneast'] = pokedex['gastrodon']
    pokedex['gastrodonwest'] = pokedex['gastrodon']

    pokedex['pikachualola'] = pokedex['pikachu']
    pokedex['pikachuoriginal'] = pokedex['pikachu']
    pokedex['pikachuhoenn'] = pokedex['pikachu']
    pokedex['pikachusinnoh'] = pokedex['pikachu']
    pokedex['pikachuunova'] = pokedex['pikachu']
    pokedex['pikachukalos'] = pokedex['pikachu']
    pokedex['pikachupartner'] = pokedex['pikachu']


    pokedex['zygardecomplete'] = pokedex['zygarde']
    pokedex['wishiwashi'] = pokedex['wishiwashischool']

    # TODO aegislash

    return pokedex


POKEDEX = _make_pokedex()