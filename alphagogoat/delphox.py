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
    NUM_HIDDEN_LAYERS = 10

    def __init__(self, input_size, hidden_layers=NUM_HIDDEN_LAYERS):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        # TODO: maybe add an encoder?
        self.rnn = nn.LSTM(input_size, Delphox.LSTM_OUTPUT_SIZE, hidden_layers)
        # self.loss = nn.L1Loss(reduction='sum')
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        move, hidden = self.rnn(x, hidden)
        return move, hidden



def make_x(turn: Battle, opponent: bool):
    pokemon = []
    moves = []

    if opponent:
        mon1 = turn.opponent_active_mon
        mon2 = turn.active_mon
        team1 = turn.opponent_team
        team2 = turn.team
    else:
        mon1 = turn.active_mon
        mon2 = turn.opponent_active_mon
        team1 = turn.team
        team2 = turn.opponent_team

    pokemon.append(EMBEDDER.embed_pokemon(mon1))
    moves.append(get_moveset(mon1))
    for t1_pokemon in team1.values():
        if t1_pokemon.species == mon1.species:
            continue
        pokemon.append(EMBEDDER.embed_pokemon(t1_pokemon).to(device=DEVICE))
        # print(pokemon[0].shape)
        moveset = get_moveset(t1_pokemon)
        # print(moveset.shape)
        moves.append(moveset)

    pokemon.append(EMBEDDER.embed_pokemon(mon2))
    moves.append(get_moveset(mon2))
    for t2_pokemon in team2.values():
        if t2_pokemon.species == mon2.species:
            continue
        pokemon.append(EMBEDDER.embed_pokemon(t2_pokemon).to(device=DEVICE))
        moveset = get_moveset(t2_pokemon)
        moves.append(moveset)
    # if opponent:
    #     mon1 = turn.opponent_active_mon
    #     mon2 = turn.active_mon
    # else:
    #     mon1 = turn.active_mon
    #     mon2 = turn.opponent_active_mon

    # pokemon = torch.hstack((EMBEDDER.embed_pokemon(mon1), EMBEDDER.embed_pokemon(mon2)))
    # moves = torch.stack((EMBEDDER.embed_moves_from_pokemon(mon1), EMBEDDER.embed_moves_from_pokemon(mon2)))

    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))
    field_conditions = EMBEDDER.embed_conditions(turn, opponent).to(device=DEVICE)
    # print(field_conditions.shape)

    x = torch.cat((pokemon, moves.flatten(), field_conditions)).unsqueeze(0)
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


def train(delphox: Delphox, data, lr=0.001, discount=0.5, weight_decay=1e-5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.Adam(delphox.parameters(), lr=lr, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True)
    total_wrong = 0
    total_correct = 0
    for turns, moves1, moves2 in data:
        hidden1_0 = (torch.zeros(Delphox.NUM_HIDDEN_LAYERS, Delphox.LSTM_OUTPUT_SIZE), torch.zeros(Delphox.NUM_HIDDEN_LAYERS, Delphox.LSTM_OUTPUT_SIZE))
        hidden2_0 = (torch.zeros(Delphox.NUM_HIDDEN_LAYERS, Delphox.LSTM_OUTPUT_SIZE), torch.zeros(Delphox.NUM_HIDDEN_LAYERS, Delphox.LSTM_OUTPUT_SIZE))
        hidden1_t = hidden1_0
        hidden2_t = hidden2_0
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag} ###")
        num_correct = 0
        num_wrong = 0
        for i, (turn, move1, move2) in enumerate(zip(turns, moves1, moves2)):
            gamma = 1 - discount / math.exp(i)

            x1 = make_x(turn, opponent=False)
            move1_pred, hidden1_t_next = delphox(x1, hidden1_t)
            move1_pred = move1_pred.squeeze(0)
            mask = get_mask(turn, opponent=False)
            move1_pred = torch.mul(move1_pred, mask)
            move1_pred = torch.where(move1_pred == 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            optimizer.zero_grad()

            type_loss = 0
            switch_loss = 0
            move1_pred_name = pred_vec_to_string(move1_pred)
            move1_name = pred_vec_to_string(move1)
            # if move1_pred_name == 'switch' and move1_name != 'switch':  # punishes predicting switch
            #     switch_loss = 1
            #     type_loss = 1
            # if move1_name != 'switch' and move1_pred_name != 'switch':
            #     type_loss = int(Move(move1_pred_name, gen=8).type != Move(move1_name, gen=8).type)

            L = gamma * (delphox.loss(move1_pred, move1) + type_loss + switch_loss)

            print(f"{turn.active_mon.species} uses {move1_pred_name} ({move1_name}) against {turn.opponent_active_mon.species}")
            print(f"loss: {L.item()}")
            L.backward(retain_graph=True)
            optimizer.step()
            if move1_pred_name == move1_name:
                num_correct += 1
            else:
                num_wrong += 1

            x2 = make_x(turn, opponent=True)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred.squeeze(0)
            mask = get_mask(turn, opponent=True)
            move2_pred = torch.mul(move2_pred, mask)
            move2_pred = torch.where(move2_pred == 0, torch.tensor(-1e10), move2_pred)
            move2_pred = F.softmax(move2_pred, dim=0)
            optimizer.zero_grad()

            # type_loss = 0
            # switch_loss = 0
            move2_pred_name = pred_vec_to_string(move2_pred)
            move2_name = pred_vec_to_string(move2)
            # if move2_pred_name == 'switch' and move2_name != 'switch':  # punishes predicting switch
            #     switch_loss = 1
            #     type_loss = 1
            # if move2_name != 'switch' and move2_pred_name != 'switch':
            #     type_loss = int(Move(move2_pred_name, gen=8).type != Move(move2_name, gen=8).type)

            L = gamma * (delphox.loss(move2_pred, move2) + type_loss + switch_loss)

            print(f"{turn.opponent_active_mon.species} uses {move2_pred_name} ({move2_name}) against {turn.active_mon.species}")
            print(f"loss: {L.item()}")
            L.backward(retain_graph=True)
            optimizer.step()
            if move2_pred_name == move2_name:
                num_correct += 1
            else:
                num_wrong += 1

        total_wrong += num_wrong
        total_correct += num_correct
        print(f"###\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"###")


def get_mask(turn: Battle, opponent: bool):
    active = turn.active_mon if not opponent else turn.opponent_active_mon
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
            move1_pred = move1_pred.squeeze(0)
            mask = get_mask(turn, opponent=False)
            move1_pred = torch.mul(move1_pred, mask)
            move1_pred = torch.where(move1_pred == 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            print(f"{turn.active_mon.species} uses {pred_vec_to_string(move1_pred)} ({pred_vec_to_string(move1)}) against {turn.opponent_active_mon.species}")
            if pred_vec_to_string(move1_pred) == pred_vec_to_string(move1):
                num_correct += 1
            else:
                num_wrong += 1

            x2 = make_x(turn, opponent=True)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred.squeeze(0)
            mask = get_mask(turn, opponent=True)
            move2_pred = torch.mul(move2_pred, mask)
            move2_pred = torch.where(move2_pred == 0, torch.tensor(-1e10), move2_pred)
            move2_pred = F.softmax(move2_pred, dim=0)
            print(f"{turn.opponent_active_mon.species} uses {pred_vec_to_string(move2_pred)} ({pred_vec_to_string(move2)}) against {turn.active_mon.species}")
            if pred_vec_to_string(move2_pred) == pred_vec_to_string(move2):
                num_correct += 1
            else:
                num_wrong += 1
        total_wrong += num_wrong
        total_correct += num_correct
        print(f"###\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"###")

