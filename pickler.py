import pickle
import json
import alphagogoat.catalogs
from poke_env.environment.battle import Battle
from alphagogoat.catalogs import MoveEnum
import torch
import re
from copy import deepcopy
from alphagogoat.embedder import process_battle, get_team_histories

