import json
import re
import time
from pathlib import Path

import requests
from joblib import Parallel, delayed
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm.auto import tqdm


def main():
    replays_url = "https://replay.pokemonshowdown.com/"
    driver = webdriver.Chrome()
    driver.get(replays_url)
    text_box = driver.find_element(by=By.NAME, value="format")
    submit_button = \
    driver.find_elements(by=By.CSS_SELECTOR, value="body > div.pfx-panel > div > form:nth-child(5) > p > button")[0]
    text_box.send_keys("gen8randombattle")
    submit_button.click()
    time.sleep(0.5)
    html = driver.page_source
    driver.close()

    # with open("Replays - Pokémon Showdown.html") as f:
    #     html = f.read()

    # make cache directory if none exists yet
    path = Path("../cache/replays/")
    path.mkdir(parents=True, exist_ok=True)

    matches = re.findall(r'href="(\/gen8randombattle-\d+)"', html)

    def cache(m):
        url = f"https://replay.pokemonshowdown.com{m}"

        # # cache .log
        # path = Path(f"../cache/replays{m}.log")
        # if not path.exists():
        #     log = requests.get(url + ".log").text
        #     with open(path, "x") as f:
        #         f.write(log)

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
