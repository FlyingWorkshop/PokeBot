class Pokemon:
    def __init__(self):
        self.name = ""
        self.species = ""
        self.gender = ""
        self.item = ""
        self.ability = ""
        self.evs = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        self.nature = ""
        self.moves = ""


class Turn:
    def __init__(self):
        self.t = 0
        self.p1_move = []
        self.p2_move = []


class Battle:
    def __init__(self):
        self.turns = []