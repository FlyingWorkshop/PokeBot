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

class Delphox(nn.Module):
    LSTM_OUTPUT_SIZE = len(MoveEnum) + 1
    NUM_HIDDEN_LAYERS = 20

    def __init__(self, input_size, hidden_layers=NUM_HIDDEN_LAYERS):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        # self.dnn = nn.Sequential(
        #     nn.Linear(input_size, input_size//2),
        #     nn.ReLU(),
        #     nn.Linear(input_size//2, input_size//2),
        #     nn.ReLU(),
        #     nn.Linear(input_size//2, input_size//2),
        #     nn.ReLU(),
        #     nn.Linear(input_size//2, len(MoveEnum) + 1),
        # )
        # TODO: maybe add an encoder?
        # self.rnn = nn.LSTM(input_size, Delphox.LSTM_OUTPUT_SIZE, hidden_layers)
        hidden_size = 5000
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, Delphox.LSTM_OUTPUT_SIZE)
        )
        self.loss = nn.L1Loss(reduction='sum')
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

def make_team(turn: Battle) -> list[Pokemon]:
    """
    Returns the current team as a list of Pokemon with the active Pokemon as the 0th element.
    """
    team = [turn.active_pokemon]
    keys = list(turn.team.keys())
    random.shuffle(keys)
    for key in keys:
        pokemon = turn.get_pokemon(key)
        if pokemon.species == turn.active_pokemon.species:
            continue
        team.append(pokemon)
    return team

def make_opponent_team(turn: Battle) -> list[Pokemon]:
    """
    Returns the current team as a list of Pokemon with the active Pokemon as the 0th element.
    """
    team = [turn.opponent_active_pokemon]
    keys = list(turn.opponent_team.keys())
    random.shuffle(keys)
    for key in keys:
        pokemon = turn.get_pokemon(key)
        if pokemon.species == turn.opponent_active_pokemon.species:
            continue
        team.append(pokemon)
    return team

def make_x(turn: Battle, opponent: bool, last_guest_correct: bool):
    pokemon = []
    moves = []

    if opponent:
        team1 = make_team(turn)
        team2 = make_opponent_team(turn)
    else:
        team1 = make_opponent_team(turn)
        team2 = make_team(turn)

    for t1_pokemon in random.sample(team1, len(team1)):
        pokemon.append(EMBEDDER.embed_pokemon(t1_pokemon).to(device=DEVICE))
        moveset = get_moveset(t1_pokemon)
        moves.append(moveset)

    for t2_pokemon in team2:
        pokemon.append(EMBEDDER.embed_pokemon(t2_pokemon).to(device=DEVICE))
        moveset = get_moveset(t2_pokemon)
        moves.append(moveset)

    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))
    field_conditions = EMBEDDER.embed_conditions(turn, opponent).to(device=DEVICE)

    mark = torch.Tensor([last_guest_correct])

    x = torch.cat((mark, pokemon, moves.flatten(), field_conditions)).unsqueeze(0)
    return x


def get_moveset(pokemon: Pokemon):
    moveset = EMBEDDER.embed_moves_from_pokemon(pokemon).to(device=DEVICE)
    possible_moves = [re.sub(r"\s|-|'", "", m.lower()) for m in sorted(POKEDEX[pokemon.species]['moves'].keys())]
    for move in pokemon.moves:
        if move in possible_moves:
            i = possible_moves.index(move)
        else:
            # TODO: handle zoroark
            break
        moveset[i, 0] = 1
    return moveset

def print_turn(turn: Battle, action1, action2):
    print(f"{turn.turn}:"
          f"\n\t{turn.active_pokemon}:\t{action1}"
          f"\n\t{turn.opponent_active_pokemon}:\t{action2}")
    print("\t[Team 1]")
    for key in turn.team:
        print(f"\t\t{turn.get_pokemon(key)}")
    print("\t[Team 2]")
    for key in turn.opponent_team:
        print(f"\t\t{turn.get_pokemon(key)}")

def apply_mask(pred, mask):
    pred = torch.mul(pred, mask)
    # pred = torch.where(pred == 0, torch.tensor(-1e10), pred)
    # pred = F.softmax(pred, dim=0)
    return pred

def train(delphox: Delphox, data, lr=0.001, discount=0.5, weight_decay=1e-5, switch_cost=100, type_cost=50):
    assert 0 <= discount <= 1
    optimizer = torch.optim.Adam(delphox.parameters(), lr=lr, weight_decay=weight_decay)
    total_wrong = 0
    total_correct = 0
    for turns, moves1, moves2 in data:
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag} ###")
        num_correct = 0
        num_wrong = 0
        last_guess_correct1 = True
        last_guess_correct2 = True
        for i, (turn, move1, move2) in enumerate(zip(turns, moves1, moves2)):

            gamma = 1 - discount / math.exp(i)

            optimizer.zero_grad()
            x1 = make_x(turn, opponent=False, last_guest_correct=last_guess_correct1)
            mask = get_mask(turn, opponent=False)
            move1_pred = delphox(x1)
            move1_pred = apply_mask(move1_pred, mask)
            L = gamma * (delphox.loss(move1_pred, move1))
            print("{:<10}->{:<10}: {:<10} ({:<10})".format(turn.active_pokemon.species, turn.opponent_active_pokemon.species, vec2str(move1_pred), {vec2str(move1)}))
            print(f"loss: {L.item()}")
            L.backward()
            optimizer.step()
            if vec2str(move1) == vec2str(move1_pred):
                num_correct += 1
                last_guess_correct1 = True
            else:
                num_wrong += 1
                last_guess_correct1 = False

            optimizer.zero_grad()
            x2 = make_x(turn, opponent=True, last_guest_correct=last_guess_correct2)
            mask = get_mask(turn, opponent=True)
            move2_pred = delphox(x2)
            move2_pred = apply_mask(move2_pred, mask)
            L = gamma * (delphox.loss(move2_pred, move1))
            print("{:<10}->{:<10}: {:<10} ({:<10})".format(turn.opponent_active_pokemon.species, turn.active_pokemon.species, vec2str(move2_pred), {vec2str(move2)}))
            print(f"loss: {L.item()}")
            L.backward()
            optimizer.step()
            if vec2str(move2) == vec2str(move2_pred):
                num_correct += 1
                last_guess_correct2 = True
            else:
                num_wrong += 1
                last_guess_correct2 = False
            
            # hidden1_t = (hidden1_t_next[0].detach(), hidden1_t_next[1].detach())
            # hidden2_t = (hidden2_t_next[0].detach(), hidden2_t_next[1].detach())

        total_wrong += num_wrong
        total_correct += num_correct

        # final_state_dict = delphox.state_dict()

        # num_sets = 0
        # for (key1, param1), (key2, param2) in zip(initial_state_dict.items(), final_state_dict.items()):
        #     num_sets += 1
        #     if not torch.equal(param1, param2):
        #         print(f'Parameter {key1} has changed')

        # print('total sets of parameters: ', num_sets)

        print(f"###\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"###")


def get_mask(turn: Battle, opponent: bool):
    active = turn.active_pokemon if not opponent else turn.opponent_active_pokemon
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

    team = turn.team if not opponent else turn.opponent_team
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
            x1 = make_x(turn, opponent=False)
            move1_pred, hidden1_t_next = delphox(x1, hidden1_t)
            move1_pred = move1_pred#.squeeze(0)
            mask = get_mask(turn, opponent=False)
            move1_pred = torch.mul(move1_pred, mask)
            move1_pred = torch.where(move1_pred == 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            print(f"{turn.active_pokemon.species}->{turn.opponent_active_pokemon.species}:\t{vec2str(move1_pred)}\t({vec2str(move1)})")
            if vec2str(move1_pred) == vec2str(move1):
                num_correct += 1
            else:
                num_wrong += 1

            x2 = make_x(turn, opponent=True)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred#.squeeze(0)
            mask = get_mask(turn, opponent=True)
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

