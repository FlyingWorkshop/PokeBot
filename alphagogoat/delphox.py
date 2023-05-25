import torch
import torch.nn as nn
from poke_env.environment.battle import Battle
import embedder
#
# from poke_env.environment.battle import Battle
#
class Delphox(nn.Module):
    def __init__(self, input_size, output_size):
        # TODO: make Delphox a RNN or LSTM; perhaps use meta-learning
        super().__init__()

        self.rnn = nn.LSTM(input_size, output_size, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, battle: Battle):

        team1, team2 = get_teams(battle)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x
