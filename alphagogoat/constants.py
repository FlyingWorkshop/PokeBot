import torch

from .catalogs import *

MAX_MOVES = 0
MAX_ITEMS = 0
MAX_ABILITIES = 0
max_items, max_abilities, max_moves = 0, 0, 0
for pokemon, data in POKEDEX.items():
    MAX_ITEMS = max(MAX_ITEMS, len(data['items']))
    MAX_ABILITIES = max(MAX_ABILITIES, len(data['abilities']))
    MAX_MOVES = max(MAX_MOVES, len(data['moves']))

BOOSTABLE_STATS = ['atk', 'def', 'spa', 'spd', 'spe']
DEFAULT_EVS = 84
EVS_PER_INC = 4
DEFAULT_IVS = 31

MOVE_EMBED_SIZE = 52
POKEMON_EMBED_SIZE = 201
NUM_POKEMON_PER_TEAM = 6
NUM_POKEMON_TOTAL = 12
NUM_CONDITIONS = 61

LSTM_INPUT_SIZE = NUM_POKEMON_TOTAL * POKEMON_EMBED_SIZE + NUM_POKEMON_TOTAL * MAX_MOVES * MOVE_EMBED_SIZE + NUM_CONDITIONS
TOTAL_POSSIBLE_MOVES = len(MoveEnum)
NUM_HIDDEN_LAYERS = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')