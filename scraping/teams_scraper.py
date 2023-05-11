import datetime
import json
from pathlib import Path

import requests


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    path = Path(f"../cache/teams/{now}.json")

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    teams_url = "https://play.pkmn.cc/data/random/gen8randombattle.json"
    response = requests.get(teams_url)
    data = response.json()

    with open(path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
