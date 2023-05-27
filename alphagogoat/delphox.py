import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon

from .constants import *
from .embedder import Embedder

EMBEDDER = Embedder()

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

    for t1_pokemon in team1.values():
        pokemon.append(EMBEDDER.embed_pokemon(t1_pokemon).to(device=device))
        moves.append(EMBEDDER.embed_moves_from_pokemon(t1_pokemon).to(device=device))

    for t2_pokemon in team2.values():
        pokemon.append(EMBEDDER.embed_pokemon(t2_pokemon).to(device=device))
        moves.append(EMBEDDER.embed_moves_from_pokemon(t2_pokemon).to(device=device))

    num_unknown_pokemon = 2 * NUM_POKEMON_PER_TEAM - len(team1) - len(team2)
    pokemon = F.pad(torch.hstack(pokemon), (0, num_unknown_pokemon * POKEMON_EMBED_SIZE), mode='constant', value=-1)
    moves = F.pad(torch.stack(moves), (0, 0, 0, 0, 0, num_unknown_pokemon))
    # TODO: add prob modification and pp updates

    field_conditions = EMBEDDER._embed_conditions(turn).to(device=device)
    x = torch.cat((pokemon, moves.flatten(), field_conditions)).unsqueeze(0)
    return x


def train(delphox: Delphox, data, lr=0.001, discount=0.5):
    assert 0 <= discount <= 1
    optimizer = torch.optim.Adam(delphox.parameters(), lr=lr)
    torch.autograd.set_detect_anomaly(True)
    for turns, history1, history2, moves1, moves2 in data:

        # TODO: have representations of the future
        
        hidden1_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))
        hidden2_0 = (torch.randn(2, Delphox.LSTM_OUTPUT_SIZE), torch.randn(2, Delphox.LSTM_OUTPUT_SIZE))

        hidden1_t = hidden1_0
        hidden2_t = hidden2_0

        for i, (turn, team1, team2, move1, move2) in enumerate(zip(turns, history1, history2, moves1, moves2)):
            
            my_active, opponent_active = turn.active_pokemon, turn.opponent_active_pokemon

            print(f"{team1=}")
            print(f"{team2=}")

            my_moves = {} if my_active.species == 'typenull' else POKEDEX[my_active.species]['moves']
            opponent_moves = {} if opponent_active.species == 'typenull' else POKEDEX[opponent_active.species]['moves']

            non_zeros_me = []
            non_zeros_opponent = []
            for m in my_moves:
                non_zeros_me.append(MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)

            for m in opponent_moves:
                non_zeros_opponent.append(MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)

            non_zeros_me = torch.tensor(non_zeros_me, dtype=torch.int64).to(device=device)
            non_zeros_opponent = torch.tensor(non_zeros_opponent, dtype=torch.int64).to(device=device)

            gamma = 1 #- discount / math.exp(i)
            x1 = make_x(turn, team1, team2)
            move1_pred, hidden1_t_next = delphox(x1, hidden1_t)
            move1_pred = move1_pred.squeeze(0)

            #print(f"{move1_pred=}")

            mask = torch.zeros_like(move1_pred).to(device=device)
            mask.scatter_(0, non_zeros_me, 1)
            move1_pred = torch.mul(move1_pred , mask)
            #print(f"{move1_pred=}")
            
            #print(f"{move1_pred=}")

            # try:
            #     move1_pred = F.softmax(move1_pred, dim=0)
            # except:
            #     print(f"{move1_pred=}")
            #     break

            move1_pred = torch.where(move1_pred <= 0, torch.tensor(-1e10), move1_pred)
            move1_pred = F.softmax(move1_pred, dim=0)
            
            optimizer.zero_grad()
            L = gamma * delphox.loss(move1_pred, move1)
            #print("Sum: ", torch.sum(move1_pred))
            #loss += L
            print(f"{L=}")
            L.backward(retain_graph=True)
            optimizer.step()

            x2 = make_x(turn, team2, team1)
            move2_pred, hidden2_t_next = delphox(x2, hidden2_t)
            move2_pred = move2_pred.squeeze(0)

            mask = torch.zeros_like(move2_pred).to(device=device)
            mask.scatter_(0, non_zeros_opponent, 1)
            move2_pred = torch.mul(move2_pred , mask)

            move2_pred = torch.where(move2_pred <= 0, torch.tensor(-1e10), move2_pred)

            move2_pred = F.softmax(move2_pred, dim=0)
            
            optimizer.zero_grad()
            loss = gamma * delphox.loss(move2_pred, move2)
            loss.backward()
            optimizer.step()
            

            # hidden1_t[0].copy_(hidden1_t_next[0])
            # hidden1_t[1].copy_(hidden1_t_next[1])
            # hidden2_t[0].copy_(hidden2_t_next[0])
            # hidden2_t[1].copy_(hidden2_t_next[1])
            # hidden1_t = (hidden1_t_next[0].clone(), hidden1_t_next[1].clone())
            # hidden2_t = (hidden2_t_next[0].clone(), hidden2_t_next[1].clone())
            #(torch.Tensor.copy_(hidden1_t_next[0]), torch.Tensor.copy_(hidden1_t_next[1]))
            #hidden2_t = (torch.Tensor.copy_(hidden2_t_next[0]), torch.Tensor.copy_(hidden2_t_next[1]))
            hidden1_t = (hidden1_t_next[0].detach(), hidden1_t_next[1].detach())
            hidden2_t = (hidden2_t_next[0].detach(), hidden2_t_next[1].detach())


    # for _ in range(reps):
    #     for battle, h1, h2, tensors_grid in tqdm(data):
    #         loss = 0
    #         hidden = (torch.randn(2, LSTM_OUTPUT_SIZE).to(device=device) , torch.randn(2, LSTM_OUTPUT_SIZE).to(device=device))
    #         for turn, team1, team2, tensor in zip(battle, h1, h2, tensors_grid):
    #             pokemon1, pokemon2 = [], []
    #             moves1, moves2 = [], []
    #
    #             for t1_pokemon in team1.values():
    #                 pokemon1.append(delphox.emb.embed_pokemon(t1_pokemon).to(device=device))
    #                 moves2.append(delphox.emb.embed_moves_from_pokemon(t1_pokemon).to(device=device))
    #
    #             for t2_pokemon in team2.values():
    #                 pokemon1.append(delphox.emb.embed_pokemon(t2_pokemon).to(device=device))
    #                 moves2.append(delphox.emb.embed_moves_from_pokemon(t2_pokemon).to(device=device))
    #
    #             tensor = tensor.to(device=device)
    #             my_active, opponent_active = turn.active_pokemon, turn.opponent_active_pokemon
    #
    #             my_moves = {} if my_active.species == 'typenull' else POKEDEX[my_active.species]['moves']
    #             opponent_moves = {} if opponent_active.species == 'typenull' else POKEDEX[opponent_active.species]['moves']
    #
    #             non_zeros = []
    #             for m in my_moves:
    #                 non_zeros.append(MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)
    #
    #             for m in opponent_moves:
    #                 non_zeros.append((TOTAL_POSSIBLE_MOVES + 1) + MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1)
    #
    #             non_zeros = torch.tensor(non_zeros, dtype=torch.int64).to(device=device)
    #
    #             for x in [a, b]:
    #             e
    #             output = output.squeeze(0)
    #             mask = torch.zeros_like(output).to(device=device)
    #             mask.scatter_(0, non_zeros, 1)
    #             output = torch.mul(output, mask)
    #             output = delphox.softmax(output)
    #             loss += delphox.loss(output, tensor)
    #
    #         optimizer.zero_grad()
    #         # TODO: fix this hacky solution on example 103/145
    #         if isinstance(loss, torch.Tensor):
    #             loss.backward()
    #         print(f"### {loss=}")
    #         optimizer.step()

# if __name__ == "__main__":
#     train(SMALL_DATASET)
