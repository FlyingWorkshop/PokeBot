import json
import logging
import copy
import re

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.effect import Effect
from poke_env.environment.move import Move, DynamaxMove
import torch


from pokedex import POKEDEX
from catalogs import Item, VolatileStatus, SIDE_COND_MAP, Ability


LOGGER = logging.getLogger('poke-env')
GEN = 8
MAX_MOVES = 8
MAX_ABILITIES = 3
MAX_ITEMS = 6
BOOSTABLE_STATS = ['atk', 'def', 'spa', 'spd', 'spe']

def process_battle(battle_json: str) -> list[Battle]:
    with open(battle_json) as f:
        battle_data = json.load(f)
    history = battle_data['log'].split('\n')
    curr_battle = Battle(battle_data['id'], battle_data['p1'], LOGGER, GEN)
    curr_battle._opponent_username = battle_data['p2']
    battle_objects = []
    for line in history:
        try:
            curr_battle._parse_message(line.split('|'))
            if line.split('|')[1] == 'turn':
                battle_objects.append(copy.deepcopy(curr_battle))
        except:
            continue

    return battle_objects


class Embedder:
    def __init__(self):
        pass

    def _embed_move(self, id: str, prob: float) -> torch.Tensor:
        """
        >>> embedder = Embedder()
        >>> embedder._embed_move("fierydance", 0).shape
        torch.Size([52])
        >>> embedder._embed_move("seismictoss", 0).shape
        torch.Size([52])
        >>> embedder._embed_move("knockoff", 0).shape
        torch.Size([52])
        >>> embedder._embed_move("leechseed", 0).shape
        torch.Size([52])
        >>> embedder._embed_move("gravapple", 0).shape
        torch.Size([52])
        >>> embedder._embed_move("appleacid", 0).shape
        torch.Size([52])
        >>> embedder._embed_move("uturn", 0).shape
        torch.Size([52])
        """
        move = Move(id, gen=8)
        embedding = [
            prob,
            move.accuracy,
            move.base_power,
            move.breaks_protect,
            move.category.value,
            move.crit_ratio,
            move.current_pp,  # TODO: how to extract this live from battle?
            move.defensive_category.value,
            move.drain,
            move.expected_hits,
            move.n_hit[0],
            move.n_hit[1],
            move.force_switch,
            move.heal,
            move.ignore_ability,
            move.ignore_defensive,
            move.ignore_evasion,
            move.is_protect_counter,
            move.is_protect_move,
            move.priority,
            move.recoil,
            1 if move.self_destruct == 'always' else 0,
            move.self_switch,
            -1 if move.side_condition is None else SideCondition[SIDE_COND_MAP[move.side_condition]].value,
            move.sleep_usable,
            move.steals_boosts,
            -1 if move.terrain is None else move.terrain.value,
            move.thaws_target,
            move.type.value,
            -1 if move.volatile_status is None else VolatileStatus[move.volatile_status].value,
            -1 if move.weather is None else move.weather.value
        ]

        # handle boosts
        if move.boosts is None:
            boosts = [0] * len(BOOSTABLE_STATS)
        else:
            boosts = []
            for stat in BOOSTABLE_STATS:
                boost = 0 if stat not in move.boosts else move.boosts[stat]
                boosts.append(boost)
        embedding += boosts

        # handle secondary effects
        secondary = []
        status = None
        secondary_boosts = None
        # on_hit = None
        volatile_status = None
        self_ = None
        for d in move.secondary:
            if 'status' in d:
                status = d
            elif 'boosts' in d:
                secondary_boosts = d
            # elif 'onHit' in d:
            #     on_hit = d
            elif 'volatileStatus' in d:
                volatile_status = d
            elif 'self' in d:
                self_ = d

        # secondary status
        if status is None:
            secondary += [0, 0]
        else:
            secondary += [status['chance'], Status[status['status'].upper()].value]

        # onHit is either "throat chop" or "anchor shot" or "tri attack", so we ignore it

        # (secondary) boosts
        if secondary_boosts is None:
            secondary += [0] * (len(BOOSTABLE_STATS) + 1)
        else:
            secondary.append(secondary_boosts['chance'])
            for stat in BOOSTABLE_STATS:
                boost = 0 if stat not in secondary_boosts['boosts'] else secondary_boosts['boosts'][stat]
                secondary.append(boost)

        # volatileStatus
        if volatile_status is None:
            secondary += [0, 0]
        else:
            secondary += [volatile_status['chance'], VolatileStatus[volatile_status['volatileStatus']].value]

        # self_
        if self_ is None:
            secondary += [0] * (len(BOOSTABLE_STATS) + 1)
        else:
            secondary.append(self_['chance'])
            for stat in BOOSTABLE_STATS:
                boost = 0 if stat not in self_['self']['boosts'] else self_['self']['boosts'][stat]
                secondary.append(boost)

        embedding += secondary

        return torch.Tensor(embedding)


    def _embed_moves_from_pokemon(self, pokemon: Pokemon):
        """
        >>> embedder = Embedder()
        >>> embedder._embed_moves_from_pokemon(Pokemon(gen=8, species="Appletun")).shape
        torch.Size([8, 52])
        >>> embedder._embed_moves_from_pokemon(Pokemon(gen=8, species="Pyukumuku")).shape
        torch.Size([8, 52])
        >>> embedder._embed_moves_from_pokemon(Pokemon(gen=8, species="Zygarde-10%")).shape
        torch.Size([8, 52])
        >>> embedder._embed_moves_from_pokemon(Pokemon(gen=8, species="Dracovish")).shape
        torch.Size([8, 52])
        >>> embedder._embed_moves_from_pokemon(Pokemon(gen=8, species="Landorus-Therian")).shape
        torch.Size([8, 52])
        """
        # make move embeddings
        embeddings = []
        moves = POKEDEX[pokemon.species]['moves']
        for name, prob in moves.items():
            id = re.sub(r"\s|-|'", "", name.lower())
            embedding = self._embed_move(id, prob)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)

        # add unknown move embeddings
        num_unknown_moves = MAX_MOVES - len(embeddings)
        embed_dim = embeddings.shape[1]
        unknown_move_embeddings = torch.full((num_unknown_moves, embed_dim), fill_value=-1)

        return torch.concat([embeddings, unknown_move_embeddings])


    def _embed_pokemon(self, pokemon: Pokemon) -> torch.Tensor:
        """
        >>> embedder = Embedder()
        >>> embedder._embed_pokemon(Pokemon(gen=8, species="Abomasnow")).shape
        torch.Size([192])
        >>> embedder._embed_pokemon(Pokemon(gen=8, species="pyukumuku")).shape
        torch.Size([192])
        >>> embedder._embed_pokemon(Pokemon(gen=8, species="dracovish")).shape
        torch.Size([192])
        >>> embedder._embed_pokemon(Pokemon(gen=8, species="accelgor")).shape
        torch.Size([192])
        """
        # abilities
        abilities = []
        for ability, prob in POKEDEX[pokemon.species]['abilities'].items():
            abilities += [prob, Ability[ability].value]
        abilities += [0] * (2 * MAX_ABILITIES - len(abilities))

        # items
        
        items = []
        for item, prob in POKEDEX[pokemon.species]['items'].items():
            items += [prob, Item[item].value]
        items += [0] * (2 * MAX_ITEMS - len(items))

        stats = pokemon.base_stats
        for stat, val in stats.items():
            if stat == 'hp':
                continue
            stats[stat] = val + 1 * pokemon.boosts[stat]
            if 'evs' in POKEDEX[pokemon.species]:
                if stat in POKEDEX[pokemon.species]['evs']:
                    stats[stat] += POKEDEX[pokemon.species]['evs'][stat]
        stats = [val for stat, val in sorted(stats.items())]

        effects = [0] * len(Effect)
        for effect in pokemon.effects:
            effects[effect.value] = 1

        status = -1 if pokemon.status is None else pokemon.status.value
        status_counter = pokemon.status_counter
        type1 = pokemon.type_1.value
        type2 = -1 if pokemon.type_2 is None else pokemon.type_2.value

        embedding = torch.Tensor(abilities + items + stats + effects + [status, status_counter, type1, type2])
        return embedding

    @staticmethod
    def get_team_histories(battles: list[Battle]):
        """
        >>> embedder = Embedder()
        >>> battles = process_battle("../cache/replays/gen8randombattle-1123651831.json")
        >>> embedder.get_team_histories(battles)

        """
        team1_history, team2_history = [], []
        team1, team2 = {}, {}
        for battle in battles:
            active = battle.active_pokemon
            opponent_active = battle.opponent_active_pokemon
            team1[active.species] = active
            team2[opponent_active.species] = opponent_active
            team1_history.append(copy.deepcopy(team1))
            team2_history.append(copy.deepcopy(team2))
        return team1_history, team2_history


