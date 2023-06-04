import random

import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon

from alphagogoat.constants import *
from alphagogoat.embedder import Embedder
from alphagogoat.utils import move_to_pred_vec_index, vec2str

import math
import random

EMBEDDER = Embedder()
POSSIBLE_ZOROARK_MOVES = sorted(POKEDEX['zoroark']['moves'].keys())
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


class Victini(nn.Module):
    def __init__(self, input_size, hidden_size=300):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x).squeeze(0)

def get_team(turn: Battle):
    team = list(turn.team.keys())
    random.shuffle(team)
    team = [turn.get_pokemon(species) for species in team]
    return team

def get_opponent_team(turn: Battle):
    team = list(turn.opponent_team.keys())
    random.shuffle(team)
    team = [turn.get_pokemon(species) for species in team]
    return team

def make_x(turn: Battle, opponent_pov: bool, last_guest_correct: bool, last_move: Move):
    if opponent_pov:
        team1 = get_team(turn)
        team2 = get_opponent_team(turn)
        pokemon1 = [EMBEDDER.embed_pokemon(mon, mon is turn.active_pokemon) for mon in team1]
        pokemon2 = [EMBEDDER.embed_pokemon(mon, mon is turn.opponent_active_pokemon) for mon in team2]
        moves1 = [EMBEDDER.embed_moves_from_pokemon(mon) for mon in team1]
        moves2 = [EMBEDDER.embed_moves_from_pokemon(mon) for mon in team2]
    else:
        team1 = get_opponent_team(turn)
        team2 = get_team(turn)
    pokemon1 = [EMBEDDER.embed_pokemon(mon, mon is turn.active_pokemon) for mon in team1]
    pokemon2 = [EMBEDDER.embed_pokemon(mon, mon is turn.opponent_active_pokemon) for mon in team2]
    moves1 = [EMBEDDER.embed_moves_from_pokemon(mon) for mon in team1]
    moves2 = [EMBEDDER.embed_moves_from_pokemon(mon) for mon in team2]

    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    pokemon = F.pad(torch.hstack(team1 + team2), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    moves = F.pad(torch.stack(moves1 + moves2), (0, 0, 0, 0, 0, num_unknown_pokemon))
    field_conditions = EMBEDDER.embed_conditions(turn, opponent_pov).to(device=DEVICE)

    mark = torch.Tensor([last_guest_correct]).to(device=DEVICE)
    last_move = EMBEDDER.embed_move(last_move, 1)

    x = torch.cat((mark, pokemon, moves.flatten(), field_conditions, last_move)).unsqueeze(0)
    return x


def apply_mask(pred, mask):
    pred = torch.mul(pred, mask)

    # Accelerock fix
    if torch.count_nonzero(pred) == 0:
        pred += mask

    # pred = torch.where(pred == 0, torch.tensor(-1e10), pred)
    # pred = F.softmax(pred, dim=0)
    return pred


def train(victini: Victini, data, lr=0.001, discount=0.5, weight_decay=1e-5, switch_cost=100, type_cost=50):
    assert 0 <= discount <= 1
    optimizer = torch.optim.Adam(victini.parameters(), lr=lr, weight_decay=weight_decay)
    total_wrong = 0
    total_correct = 0
    for turns, moves1, moves2 in data:
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag} ###")
        num_correct = 0
        num_wrong = 0
        last_guess_correct1 = True
        last_move1 = None
        last_guess_correct2 = True
        last_move2 = None
        for i, (turn, move1, move2) in enumerate(zip(turns, moves1, moves2)):
            gamma = 1 - discount / math.exp(i)
            optimizer.zero_grad()
            x1 = make_x(turn, opponent_pov=False, last_guest_correct=last_guess_correct1, last_move=last_move1)
            mask = get_mask(turn, opponent_pov=False)
            move1_pred = victini(x1)
            move1_pred = apply_mask(move1_pred, mask)
            L = gamma * (victini.loss(move1_pred, move1))
            if vec2str(move1) == vec2str(move1_pred):
                num_correct += 1
                last_guess_correct1 = True
                color = GREEN
            else:
                num_wrong += 1
                last_guess_correct1 = False
                color = RED
            print(color + "{:<30} {:<30} {:<30} {:<30}".format(turn.active_pokemon.species, turn.opponent_active_pokemon.species, vec2str(move1_pred), vec2str(move1)) + RESET)
            print(f"loss: {L.item()}")
            L.backward()


            optimizer.zero_grad()
            x2 = make_x(turn, opponent_pov=True, last_guest_correct=last_guess_correct2, last_move=last_move2)
            mask = get_mask(turn, opponent_pov=True)
            move2_pred = victini(x2)
            move2_pred = apply_mask(move2_pred, mask)
            L = gamma * (victini.loss(move2_pred, move2))
            if vec2str(move2) == vec2str(move2_pred):
                num_correct += 1
                last_guess_correct2 = True
                color = GREEN
            else:
                num_wrong += 1
                last_guess_correct2 = False
                color = RED
            print(color + "{:<30} {:<30} {:<30} {:<30}".format(turn.opponent_active_pokemon.species, turn.active_pokemon.species, vec2str(move2_pred), vec2str(move2)) + RESET)
            print(f"loss: {L.item()}")
            L.backward()
            optimizer.step()


        total_wrong += num_wrong
        total_correct += num_correct

        print(f"###\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"###")


def get_mask(turn: Battle, opponent_pov: bool):
    active = turn.active_pokemon if not opponent_pov else turn.opponent_active_pokemon
    moves = POKEDEX[active.species]['moves']
    predicted_move_indices = []
    seen_moves = active.moves
    is_zoroark = any(m not in moves for m in seen_moves)
    if len(seen_moves) == 4 and not is_zoroark:
        for s, m in seen_moves.items():
            if m.current_pp > 0:
                predicted_move_indices.append(move_to_pred_vec_index(s))
    else:
        for m in moves:
            if m in seen_moves:
                if seen_moves[m].current_pp > 0:
                    # seen move out of PP
                    predicted_move_indices.append(move_to_pred_vec_index(m))
            else:
                # unseen but possible move
                predicted_move_indices.append(move_to_pred_vec_index(m))

    team = turn.team if not opponent_pov else turn.opponent_team
    num_fainted = sum(mon.fainted for mon in team.values())
    if num_fainted < 5:  # can switch
        predicted_move_indices.append(len(MoveEnum))
    predicted_move_indices = torch.tensor(predicted_move_indices, dtype=torch.int64).to(device=DEVICE)
    mask = torch.zeros(Delphox.LSTM_OUTPUT_SIZE).to(device=DEVICE)
    mask.scatter_(0, predicted_move_indices, 1)
    return mask


def evaluate(delphox, data):
    total_correct = 0
    total_wrong = 0
    for turns, moves1, moves2 in data:
        hidden1_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))
        hidden2_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))

        hidden1_t = hidden1_0
        hidden2_t = hidden2_0

        num_correct = 0
        num_wrong = 0
        for i, (turn, move1, move2) in enumerate(zip(turns, moves1, moves2)):
            x1 = make_x(turn, opponent_pov=False)
            move1_pred, hidden1_t_next = delphox(x1, hidden1_t)
            move1_pred = move1_pred#.squeeze(0)
            mask = get_mask(turn, opponent_pov=False)
            move1_pred = torch.mul(move1_pred, mask)
            move1_pred = torch.where(move1_pred == 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            print(f"{turn.active_pokemon.species}->{turn.opponent_active_pokemon.species}:\t{vec2str(move1_pred)}\t({vec2str(move1)})")
            if vec2str(move1_pred) == vec2str(move1):
                num_correct += 1
            else:
                num_wrong += 1

            x2 = make_x(turn, opponent_pov=True)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred
            mask = get_mask(turn, opponent_pov=True)
            move2_pred = torch.mul(move2_pred, mask)
            move2_pred = torch.where(move2_pred == 0, torch.tensor(-1e10), move2_pred)
            move2_pred = F.softmax(move2_pred, dim=0)
            print(f"{turn.opponent_active_pokemon.species}->{turn.active_pokemon.species}:\t{vec2str(move2_pred)}\t({vec2str(move2)})")
            if vec2str(move2_pred) == vec2str(move2):
                num_correct += 1
            else:
                num_wrong += 1

            hidden1_t = (hidden1_t_next[0].detach(), hidden1_t_next[1].detach())
            hidden2_t = (hidden2_t_next[0].detach(), hidden2_t_next[1].detach())

        total_wrong += num_wrong
        total_correct += num_correct
        print(f"###\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"###")

