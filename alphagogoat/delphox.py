import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon

from .constants import *
from .embedder import Embedder
from .utils import move_to_pred_vec_index, pred_vec_to_string

import math

EMBEDDER = Embedder()
POSSIBLE_ZOROARK_MOVES = sorted(POKEDEX['zoroark']['moves'].keys())

class Delphox(nn.Module):
    LSTM_OUTPUT_SIZE = len(MoveEnum) + 1

    def __init__(self, input_size, hidden_layers=2):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        # TODO: maybe add an encoder?
        self.rnn = nn.LSTM(input_size, Delphox.LSTM_OUTPUT_SIZE, hidden_layers)
        #self.softmax = nn.Softmax(dim=0)
        self.loss = nn.L1Loss(reduction='sum')

    def forward(self, x, hidden):
        move, hidden = self.rnn(x, hidden)
        return move, hidden



def make_x(turn: Battle, team1: dict[str: Pokemon], team2: dict[str: Pokemon]):
    pokemon = []
    moves = []

    for species, t1_pokemon in team1.items():
        pokemon.append(EMBEDDER.embed_pokemon(t1_pokemon).to(device=DEVICE))
        moveset = get_moveset(species, t1_pokemon)
        moves.append(moveset)

    for species, t2_pokemon in team2.items():
        pokemon.append(EMBEDDER.embed_pokemon(t2_pokemon).to(device=DEVICE))
        moveset = get_moveset(species, t2_pokemon)
        moves.append(moveset)

    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))
    field_conditions = EMBEDDER.embed_conditions(turn).to(device=DEVICE)

    x = torch.cat((pokemon, moves.flatten(), field_conditions)).unsqueeze(0)
    return x


def get_moveset(species, moves):
    moveset = EMBEDDER.embed_moves_from_pokemon(moves).to(device=DEVICE)
    possible_moves = [re.sub(r"\s|-|'", "", m.lower()) for m in sorted(POKEDEX[species]['moves'].keys())]
    for move in moves.moves:
        if move in possible_moves:
            i = possible_moves.index(move)
        else:
            # TODO: handle zoroark
            break
        moveset[i, 0] = 1
    return moveset


def train(delphox: Delphox, data, lr=0.001, discount=0.5, weight_decay=1e-5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.Adam(delphox.parameters(), lr=lr, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True)
    for turns, history1, history2, moves1, moves2 in data:
        hidden1_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))
        hidden2_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))
        hidden1_t = hidden1_0
        hidden2_t = hidden2_0
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag}")
        for i, (turn, team1, team2, move1, move2) in enumerate(zip(turns, history1, history2, moves1, moves2)):
            print(f"Turn {i+1}")
            gamma = 1 - discount / math.exp(i)
            for pokemon in turn.all_active_pokemons:
                print(f"{pokemon.species}: \n\t{pokemon.current_hp=}\n\t{pokemon.current_hp_fraction=}\n\t{pokemon.fainted=}")
                print(f"{turn.available_switches=}")
                print(f"{turn.fields=}")
                print(f"{turn.weather=}")
                print(f"{turn.fainted=}")
                print(f"{turn.team=}")
                print(f"{turn.opponent_team}")

            x1 = make_x(turn, team1, team2)
            move1_pred, hidden1_t_next = delphox(x1, hidden1_t)
            move1_pred = move1_pred.squeeze(0)
            mask = get_mask(move1_pred, team1, turn.active_pokemon, turn.fainted)
            move1_pred = torch.mul(move1_pred, mask)
            move1_pred = torch.where(move1_pred == 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            optimizer.zero_grad()
            L = gamma * delphox.loss(move1_pred, move1)
            print(f"{turn.active_pokemon.species} uses {pred_vec_to_string(move1_pred)} ({pred_vec_to_string(move1)}) against {turn.opponent_active_pokemon.species}")
            print(f"loss: {L.item()}")
            L.backward(retain_graph=True)
            optimizer.step()

            x2 = make_x(turn, team2, team1)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred.squeeze(0)
            mask = get_mask(move2_pred, team2, turn.opponent_active_pokemon, turn.opponent_fainted)
            move2_pred = torch.mul(move2_pred, mask)
            move2_pred = torch.where(move2_pred == 0, torch.tensor(-1e10), move2_pred)
            move2_pred = F.softmax(move2_pred, dim=0)
            optimizer.zero_grad()
            L = gamma * delphox.loss(move2_pred, move2)
            print(f"{turn.opponent_active_pokemon.species} uses {pred_vec_to_string(move2_pred)} ({pred_vec_to_string(move2)}) against {turn.active_pokemon.species}")
            print(f"loss: {L.item()}")
            L.backward(retain_graph=True)
            optimizer.step()


def get_mask(move_pred, team: dict[str: Pokemon], active: Pokemon, fainted):
    moves = POKEDEX[active.species]['moves']
    predicted_move_indices = []
    seen_moves = team[active.species].moves
    is_zoroark = any(m not in moves for m in seen_moves)
    if len(seen_moves) == 4 and not is_zoroark:
        for m in seen_moves:
            predicted_move_indices.append(move_to_pred_vec_index(m))
    else:
        for m in moves:
            predicted_move_indices.append(move_to_pred_vec_index(m))
    if len(fainted) < 5:
        predicted_move_indices.append(len(MoveEnum))
    predicted_move_indices = torch.tensor(predicted_move_indices, dtype=torch.int64).to(device=DEVICE)
    mask = torch.zeros_like(move_pred).to(device=DEVICE)
    mask.scatter_(0, predicted_move_indices, 1)
    return mask


def evaluate(delphox, data):
    total_correct = 0
    total_wrong = 0
    for turns, history1, history2, moves1, moves2 in data:
        hidden1_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))
        hidden2_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))

        hidden1_t = hidden1_0
        hidden2_t = hidden2_0

        num_correct = 0
        num_wrong = 0
        for i, (turn, team1, team2, move1, move2) in enumerate(zip(turns, history1, history2, moves1, moves2)):
            x1 = make_x(turn, team1, team2)
            move1_pred, hidden1_t_next = delphox(x1, hidden1_t)
            move1_pred = move1_pred.squeeze(0)
            mask = get_mask(move1_pred, team1, turn.active_pokemon)
            move1_pred = torch.mul(move1_pred, mask)
            move1_pred = torch.where(move1_pred == 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            print(f"{turn.active_pokemon.species} uses {pred_vec_to_string(move1_pred)} ({pred_vec_to_string(move1)}) against {turn.opponent_active_pokemon.species}")
            if pred_vec_to_string(move1_pred) == {pred_vec_to_string(move1)}:
                num_correct += 1
            else:
                num_wrong += 1

            x2 = make_x(turn, team2, team1)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred.squeeze(0)
            mask = get_mask(move2_pred, team2, turn.opponent_active_pokemon)
            move2_pred = torch.mul(move2_pred, mask)
            move2_pred = torch.where(move2_pred == 0, torch.tensor(-1e10), move2_pred)
            move2_pred = F.softmax(move2_pred, dim=0)
            print(
                f"{turn.opponent_active_pokemon.species} uses {pred_vec_to_string(move2_pred)} ({pred_vec_to_string(move2)}) against {turn.active_pokemon.species}")
            if pred_vec_to_string(move2_pred) == {pred_vec_to_string(move2)}:
                num_correct += 1
            else:
                num_wrong += 1
            total_wrong += num_wrong
            total_correct += num_correct
        print(f"###\nbattle accuracy:\t{num_correct / (num_correct + num_wrong)}\noverall accuracy:\t{total_correct / (total_correct + total_wrong)}\n###")

