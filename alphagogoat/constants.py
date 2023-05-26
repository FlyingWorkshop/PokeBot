from catalogs import *
import torch

MAX_NUM_POSSIBLE_MOVES = 8
MOVE_EMBED_SIZE = 52
POKEMON_EMBED_SIZE = 192
NUM_POKEMON_PER_TEAM = 6
NUM_POKEMON_TOTAL = 12

LSTM_INPUT_SIZE = NUM_POKEMON_TOTAL * POKEMON_EMBED_SIZE + NUM_POKEMON_TOTAL * MAX_NUM_POSSIBLE_MOVES * MOVE_EMBED_SIZE

TOTAL_POSSIBLE_MOVES = len(MoveEnum)

LSTM_OUTPUT_SIZE = 2 * (TOTAL_POSSIBLE_MOVES + 1)

NUM_HIDDEN_LAYERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')