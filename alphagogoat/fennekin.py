import torch
import torch.nn as nn
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon


from pokedex import POKEDEX


class Fennekin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        # embed battle
        # weather = turn.weather
        # if is_p1:
        #     mon1 = turn.active_pokemon
        #     team1 = turn.team
        #     side1 = turn.side_conditions
        #     mon2 = turn.opponent_active_pokemon
        #     team2 = turn.opponent_team
        #     side2 = turn.opponent_side_conditions
        # else:
        #     mon1 = turn.opponent_active_pokemon
        #     team1 = turn.opponent_active_pokemon
        #     side1 = turn.opponent_side_conditions
        #     mon2 = turn.active_pokemon
        #     team2 = turn.team
        #     side2 = turn.side_conditions



def train(fennekin: Fennekin, data):
    for turns, actions1, actions2 in data:
        for turn, action1, action2 in zip(turns, actions1, actions2):
            # embed battle


            # embed damage

            # make damage cube
    pass


