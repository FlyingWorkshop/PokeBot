import json
import re


def _get_turns(log) -> list:
    turns = []
    turn = []
    for line in log.split('\n'):
        if line.startswith("|turn|") and turn:
            turns.append(turn)
            turn = []
        turn.append(line)
    return turns


def make_battle(battle_json, team_json):
    with open(battle_json) as f:
        battle_data = json.load(f)
    with open(team_json) as f:
        team_data = json.load(f)

    return _get_turns(battle_data['log'])