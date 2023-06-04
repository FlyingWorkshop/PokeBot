import logging
import random
from pathlib import Path

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from grave.fennekin import Fennekin, train, evaluate
from alphagogoat.utils import make_data

LOGGER = logging.getLogger('poke-env')

def main():
    """
    TODO: ideas
    - include ELO in embedding (the model should have different predictions for skilled and unskilled players)
    - less penalty for guessing the wrong move but correct type, category (TODO: make this a damage multiplier difference instead)
    - heavier penalty for guessing switching incorrectly
    - make delphox deeper
    """
    # delphox_path = "fennekin.pth"
    json_files = [filepath for filepath in Path("../cache/replays").iterdir() if filepath.name.endswith('.json')]
    train_files, test_files = json_files[:-10], json_files[-10:]
    reps = 10000
    fennekin = Fennekin()
    for _ in range(reps):
        random.shuffle(train_files)
        # train_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(train_files))
        # train_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(train_files[:100]))  # MEDIUM
        train_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(train_files[:30]))  # SMALL
        # train_data = [make_data(f) for f in tqdm(json_files[:1])]  # SINGLE-PROCESS DEBUGGING
        # train_data = [make_data("cache/replays/gen8randombattle-1872565566.json")]
        # if Path(delphox_path).exists():
        #     fennekin.load_state_dict(torch.load(delphox_path))
        #     fennekin.eval()
        train(fennekin, train_data, lr=100)
    # torch.save(delphox.state_dict(), delphox_path)
    test_data = Parallel(n_jobs=4)(delayed(make_data)(filepath) for filepath in tqdm(test_files))
    evaluate(fennekin, test_data)



if __name__ == "__main__":
    main()

