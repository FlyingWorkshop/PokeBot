import json
import re
import time
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By


def main():
    with open("Replays - Pokémon Showdown.html") as f:
        html = f.read()

    # make cache directory if none exists yet
    path = Path("../cache/replays/")
    path.mkdir(parents=True, exist_ok=True)

    matches = re.findall(r'href="(\/gen8randombattle-\d+)"', html)

    def cache(m):
        url = f"https://replay.pokemonshowdown.com{m}"

        # cache .log
        path = Path(f"../cache/replays{m}.log")
        if not path.exists():
            log = requests.get(url + ".log").text
            with open(path, "x") as f:
                f.write(log)

        # cache .json
        path = Path(f"../cache/replays{m}.json")
        if not path.exists():
            response = requests.get(url + ".json")
            data = response.json()
            with open(path, "x") as f:
                json.dump(data, f)

    Parallel(n_jobs=4)(delayed(cache)(m) for m in tqdm(matches))


if __name__ == "__main__":
    main()
