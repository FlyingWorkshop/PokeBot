import torch
import torch.nn as nn
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move


from alphagogoat.pokedex import POKEDEX
from alphagogoat.catalogs import MoveEnum
from alphagogoat.constants import MAX_MOVES, NUM_POKEMON_PER_TEAM


def get_possible_moves(pokemon: Pokemon):
    return list(sorted(POKEDEX[pokemon.species]['moves']))

class Fennekin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: add multihead attention
        input_size = NUM_POKEMON_PER_TEAM * NUM_POKEMON_PER_TEAM * MAX_MOVES
        hidden_size = 4
        num_classes = MAX_MOVES
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=0)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, active: Pokemon, x: torch.Tensor):
        pred = self.model(x.flatten())
        mask = torch.zeros(MAX_MOVES)
        mask[:len(get_possible_moves(active))] = 1
        pred = pred * mask
        return pred

def print_turn(turn: Battle, action1, action2):
    print(f"{turn.turn}:"
          f"\n\t{turn.active_pokemon}:\t{action1}"
          f"\n\t{turn.opponent_active_pokemon}:\t{action2})")
    print("\t[Team 1]")
    for key in turn.team:
        print(f"\t\t{turn.get_pokemon(key)}")
    print("\t[Team 2]")
    for key in turn.opponent_team:
        print(f"\t\t{turn.get_pokemon(key)}")

def make_team(turn: Battle) -> list[Pokemon]:
    """
    Returns the current team as a list of Pokemon with the active Pokemon as the 0th element.
    """
    team = [turn.active_pokemon]
    for key in turn.team:
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
    for key in turn.opponent_team:
        pokemon = turn.get_pokemon(key)
        if pokemon.species == turn.opponent_active_pokemon.species:
            continue
        team.append(pokemon)
    return team

def embed_damage(team1: list[Pokemon], team2: list[Pokemon]):
    damage = -torch.ones((NUM_POKEMON_PER_TEAM, NUM_POKEMON_PER_TEAM, MAX_MOVES))
    for i, mon1 in enumerate(team1):
        for j, mon2 in enumerate(team2):
            for k, move in enumerate(sorted(POKEDEX[mon1.species]['moves'])):
                move = Move(move, gen=8)
                damage[i, j, k] = move.base_power / 100
                if move.base_power > 0:
                    damage[i, j, k] *= mon2.damage_multiplier(move)
    return damage

def make_label(turn: Battle, action: tuple[str]):
    if action[0] == 'move':
        move = action[1]
        nondynamax_moves = sorted(list(POKEDEX[turn.active_pokemon.species]['moves']))
        if move in nondynamax_moves:
            i = sorted(list(POKEDEX[turn.active_pokemon.species]['moves'])).index(move)
            label = torch.zeros(MAX_MOVES)
            label[i] = 1
            return label
    # TODO: handle switches and dynamax
    return torch.zeros(MAX_MOVES)



def train(fennekin: Fennekin, data: list[list[Battle], list[tuple], list[tuple]], lr=0.001):
    opt = torch.optim.SGD(fennekin.parameters(), lr=lr)
    for turns, actions1, actions2 in data:
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag} ###")
        for turn, action1, action2 in zip(turns, actions1, actions2):

            # print_turn(turn, action1, action2)
            print(f"Turn {turn.turn}")

            # embed damage
            team1 = make_team(turn)
            team2 = make_opponent_team(turn)

            damage = embed_damage(team1, team2)
            pred = fennekin(turn.active_pokemon, damage)

            # make label
            opt.zero_grad()
            label = make_label(turn, action1)
            loss = fennekin.loss(pred, label)
            print(f"\tloss: {loss.item()}")
            loss.backward()
            opt.step()

            # print
            i = torch.argmax(pred).item()
            pred1 = sorted(list(POKEDEX[turn.active_pokemon.species]['moves']))[i]
            print(f"\t{turn.active_pokemon.species} uses {action1} ({pred1}) against {turn.opponent_active_pokemon.species}")

def evaluate(fennekin: Fennekin,  data: list[list[Battle], list[tuple], list[tuple]]):
    total_correct = 0
    total_wrong = 0
    guesses = [0 for _ in range(MAX_MOVES)]
    for turns, actions1, actions2 in data:
        print(f"### https://replay.pokemonshowdown.com/{turns[0].battle_tag} ###")
        num_correct, num_wrong = 0, 0
        for turn, action1, action2 in zip(turns, actions1, actions2):
            # print_turn(turn, action1, action2)
            print(f"Turn {turn.turn}")

            # embed damage
            team1 = make_team(turn)
            team2 = make_opponent_team(turn)

            damage = embed_damage(team1, team2)
            pred = fennekin(turn.active_pokemon, damage)

            # print
            i = torch.argmax(pred).item()
            guesses[i] += 1
            pred1 = sorted(list(POKEDEX[turn.active_pokemon.species]['moves']))[i]
            print(f"\t{turn.active_pokemon.species} uses {action1} ({pred1}) against {turn.opponent_active_pokemon.species}")
            if pred1 == actions1[1]:
                num_correct += 1
            else:
                num_wrong += 1

        total_wrong += num_wrong
        total_correct += num_correct
        print(f"###\n"
              f"guesses:\t{guesses}\n"
              f"battle accuracy:\t{num_correct / (num_correct + num_wrong + 1e-10)}\n"
              f"overall accuracy:\t{total_correct / (total_correct + total_wrong + 1e-10)}\n"
              f"###")
