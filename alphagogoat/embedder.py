"""
NOTE: doctests are outdated!

# TODO: ask Adam if we handle battle.maybe_trapped, battle.can_dynamx, etc. (battle attributes)
"""
import copy
import json
import logging
import re
import random

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.effect import Effect
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field

from .catalogs import Item, VolatileStatus, SIDE_COND_MAP, Ability, MoveEnum
from .pokedex import POKEDEX
# from pokedex import POKEDEX
# from catalogs import Item, VolatileStatus, SIDE_COND_MAP, Ability, MoveEnum

from .constants import MAX_MOVES,MAX_ABILITIES, MAX_ITEMS, BOOSTABLE_STATS, DEFAULT_EVS, DEFAULT_IVS, EVS_PER_INC


def process_battle(battle_json: str) -> list[Battle]:
    """
    >>> battles = process_battle("../cache/replays/gen8randombattle-1123651831.json")

    """
    with open(battle_json) as f:
        battle_data = json.load(f)

    log = battle_data['log'].split('\n')
    curr_battle = Battle(battle_data['id'], battle_data['p1'], logging.getLogger('poke-env'), 8)
    curr_battle._opponent_username = battle_data['p2']
    battle_objects = []
    for line in log:
        # TODO: get rid of this try block!
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

    @staticmethod
    def embed_conditions(battle: Battle, opponent: bool) -> torch.FloatTensor:
        # takes in a battle object, and returns a tensor filled with the 
        # side conditions of both sides, field, and weather of both sides
        d1 = battle.opponent_side_conditions if opponent else battle.side_conditions
        d2 = battle.side_conditions if opponent else battle.opponent_side_conditions

        embed1 = [
            -1 if SideCondition['LIGHT_SCREEN'] not in d1 else d1[SideCondition['LIGHT_SCREEN']],
            -1 if SideCondition['REFLECT'] not in d1 else d1[SideCondition['REFLECT']],
            -1 if SideCondition['SPIKES'] not in d1 else d1[SideCondition['SPIKES']],
            -1 if SideCondition['STEALTH_ROCK'] not in d1 else d1[SideCondition['STEALTH_ROCK']],
            -1 if SideCondition['STICKY_WEB'] not in d1 else d1[SideCondition['STICKY_WEB']],
            -1 if SideCondition['TAILWIND'] not in d1 else d1[SideCondition['TAILWIND']],
            -1 if SideCondition['TOXIC_SPIKES'] not in d1 else d1[SideCondition['TOXIC_SPIKES']],
        ]

        embed2 = [
            -1 if SideCondition['LIGHT_SCREEN'] not in d2 else d2[SideCondition['LIGHT_SCREEN']],
            -1 if SideCondition['REFLECT'] not in d2 else d2[SideCondition['REFLECT']],
            -1 if SideCondition['SPIKES'] not in d2 else d2[SideCondition['SPIKES']],
            -1 if SideCondition['STEALTH_ROCK'] not in d2 else d2[SideCondition['STEALTH_ROCK']],
            -1 if SideCondition['STICKY_WEB'] not in d2 else d2[SideCondition['STICKY_WEB']],
            -1 if SideCondition['TAILWIND'] not in d2 else d2[SideCondition['TAILWIND']],
            -1 if SideCondition['TOXIC_SPIKES'] not in d2 else d2[SideCondition['TOXIC_SPIKES']],
        ]

        weather = [0, 0]
        if battle.weather:
            weather = list(list(battle.weather.items())[0])
            weather[0] = weather[0].value
        embed = embed1 + embed2 + weather

        # embed = []
        # for side_condition in SideCondition:
        #     if side_condition in battle.side_conditions and battle.side_conditions[side_condition]:
        #         embed.append(battle.side_conditions[side_condition])
        #     else:
        #         embed.append(0)
        #
        # for side_condition in SideCondition:
        #     if side_condition in battle.opponent_side_conditions and battle.opponent_side_conditions[side_condition]:
        #         embed.append(battle.opponent_side_conditions[side_condition])
        #     else:
        #         embed.append(0)
        #
        # for w in Weather:
        #     if w in battle.weather and battle.weather[w]:
        #         embed.append(battle.weather[w])
        #     else:
        #         embed.append(0)
        #
        # for f in Field:
        #     if f in battle.fields and battle.fields[f]:
        #         embed.append(battle.fields[f])
        #     else:
        #         embed.append(0)
        
        return torch.FloatTensor(embed)


            
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
            # move.breaks_protect,
            move.category.value,
            # move.crit_ratio,
            # move.current_pp,
            # move.defensive_category.value,
            move.drain,
            move.expected_hits,
            # move.n_hit[0],
            # move.n_hit[1],
            # move.force_switch,
            move.heal,
            # move.ignore_ability,
            # move.ignore_defensive,
            # move.ignore_evasion,
            # move.is_protect_counter,
            move.is_protect_move,
            move.priority,
            # move.recoil,
            1 if move.self_destruct == 'always' else 0,
            move.self_switch,
            -1 if move.side_condition is None else SideCondition[SIDE_COND_MAP[move.side_condition]].value,
            move.sleep_usable,
            # -1 if move.terrain is None else move.terrain.value,
            # move.thaws_target,
            move.type.value,
            -1 if move.volatile_status is None else VolatileStatus[move.volatile_status].value,
            # -1 if move.weather is None else move.weather.value
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

        # # handle secondary effects
        # secondary = []
        # status = None
        # secondary_boosts = None
        # # on_hit = None
        # volatile_status = None
        # self_ = None
        # for d in move.secondary:
        #     if 'status' in d:
        #         status = d
        #     elif 'boosts' in d:
        #         secondary_boosts = d
        #     # elif 'onHit' in d:
        #     #     on_hit = d
        #     elif 'volatileStatus' in d:
        #         volatile_status = d
        #     elif 'self' in d:
        #         self_ = d
        #
        # # secondary status
        # if status is None:
        #     secondary += [0, 0]
        # else:
        #     secondary += [status['chance'], Status[status['status'].upper()].value]
        #
        # # onHit is either "throat chop" or "anchor shot" or "tri attack", so we ignore it
        #
        # # (secondary) boosts
        # if secondary_boosts is None:
        #     secondary += [0] * (len(BOOSTABLE_STATS) + 1)
        # else:
        #     secondary.append(secondary_boosts['chance'])
        #     for stat in BOOSTABLE_STATS:
        #         boost = 0 if stat not in secondary_boosts['boosts'] else secondary_boosts['boosts'][stat]
        #         secondary.append(boost)
        #
        # # volatileStatus
        # if volatile_status is None:
        #     secondary += [0, 0]
        # else:
        #     secondary += [volatile_status['chance'], VolatileStatus[volatile_status['volatileStatus']].value]
        #
        # # self_
        # if self_ is None:
        #     secondary += [0] * (len(BOOSTABLE_STATS) + 1)
        # else:
        #     secondary.append(self_['chance'])
        #     for stat in BOOSTABLE_STATS:
        #         boost = 0 if stat not in self_['self']['boosts'] else self_['self']['boosts'][stat]
        #         secondary.append(boost)
        #
        # embedding += secondary

        return torch.Tensor(embedding)

    def embed_moves_from_pokemon(self, pokemon: Pokemon):
        """
        >>> embedder = Embedder()
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Appletun")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Pyukumuku")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Zygarde-10%")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Dracovish")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Landorus-Therian")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Cinderace")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Solrock")).shape
        torch.Size([8, 52])
        >>> embedder.embed_moves_from_pokemon(Pokemon(gen=8, species="Type: Null")).shape
        torch.Size([8, 52])
        """
        # make move embeddings
        embeddings = []
        moves = POKEDEX[pokemon.species]['moves']
        # for name, prob in sorted(moves.items()):
        items_list = list(moves.items())
        
        for name, prob in random.sample(items_list, len(items_list)):
            id = re.sub(r"\s|-|'", "", name.lower())
            embedding = self._embed_move(id, prob)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)

        # add unknown move embeddings
        num_unknown_moves = MAX_MOVES - len(embeddings)
        embed_dim = embeddings.shape[1]
        unknown_move_embeddings = torch.full((num_unknown_moves, embed_dim), fill_value=-1)

        return torch.concat([embeddings, unknown_move_embeddings])

    @staticmethod
    def embed_pokemon(pokemon: Pokemon) -> torch.Tensor:
        """
        >>> embedder = Embedder()
        >>> battles = process_battle("../cache/replays/gen8randombattle-1123651831.json")
        >>> embedder.embed_pokemon(battles[0].active_pokemon).shape
        torch.Size([195])
        >>> embedder.embed_pokemon(Pokemon(species='Solrock', gen=8)).shape
        torch.Size([195])
        """
        # TODO: feature reduction
        # TODO:add more flags and data (is dynamaxed, level, preparing, weight)

        data = POKEDEX[pokemon.species]
        embedding = [
            pokemon.current_hp or 0,  # handles when current_hp is None
            # pokemon.first_turn,
            pokemon.is_dynamaxed,
            # pokemon.level,
            # pokemon.must_recharge or pokemon.preparing,  # TODO: handle recharge and preparing
            # pokemon.protect_counter,
            pokemon.type_1.value,
            -1 if pokemon.type_2 is None else pokemon.type_2.value,
            -1 if pokemon.status is None else pokemon.status.value,
            # pokemon.status_counter,
            # pokemon.weight,
        ]

        # abilities
        # abilities = []
        # if pokemon.ability == 'unknown_ability' or pokemon.ability is None:
        #     for ability, prob in data['abilities'].items():
        #         abilities += [prob, Ability[ability].value]
        # else:
        #     abilities += [1, Ability[pokemon.ability].value]
        # abilities += [0] * (2 * MAX_ABILITIES - len(abilities))

        # items
        # TODO: handle knocked off items
        items = []
        if pokemon.item == 'unknown_item' or pokemon.item is None:
            for item, prob in data['items'].items():
                items += [prob, Item[item].value]
        else:
            items += [1, Item[pokemon.item].value]
        items += [0] * (2 * MAX_ITEMS - len(items))

        # effects
        # effects = [0] * len(Effect)
        # for effect in pokemon.effects:
        #     effects[effect.value - 1] = 1
        effects = [
            Effect['CONFUSION'] in pokemon.effects,
            Effect['ENCORE'] in pokemon.effects,
            # Effect['FLASH_FIRE'] in pokemon.effects,
            any(e in pokemon.effects for e in (Effect['FIRE_SPIN'], Effect['TRAPPED'], Effect['MAGMA_STORM'], Effect['WHIRLPOOL'])),
            Effect['LEECH_SEED'] in pokemon.effects,
            # Effect['STICKY_WEB'] in pokemon.effects,
            Effect['SUBSTITUTE'] in pokemon.effects,
            Effect['YAWN'] in pokemon.effects,
            Effect['NO_RETREAT'] in pokemon.effects,
            Effect['MAGNET_RISE'] in pokemon.effects
        ]

        # stats
        stats = pokemon.base_stats.copy()
        for stat, val in stats.items():
            if stat != 'hp':
                stats[stat] = val + 1 * pokemon.boosts[stat]
            if 'evs' in data:
                evs = DEFAULT_EVS
                if stat in data['evs']:
                    evs += data['evs'][stat]
                stats[stat] = round(stats[stat] + evs // EVS_PER_INC + pokemon.level * DEFAULT_IVS)
        stats = [val for stat, val in sorted(stats.items())]

        # embedding = torch.Tensor(abilities + items + stats + effects + embedding)
        embedding = torch.Tensor(items + stats + effects + embedding)
        return embedding


def get_team_histories(battles: list[Battle]):
    """
    >>> battles = process_battle("../cache/replays/gen8randombattle-1123651831.json")
    >>> get_team_histories(battles)
    """
    team1_history, team2_history = [], []
    team1, team2 = {}, {}
    for battle in battles:
        active = battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon
        team1[active.species] = active
        team2[opponent_active.species] = opponent_active
        if active.species not in team1:
            team1_history.append(copy.deepcopy(team1))
        else:
            team1_history.append(team1)
        if opponent_active.species not in team2:
            team2_history.append(copy.deepcopy(team2))
        else:
            team2_history.append(team2)
    return team1_history, team2_history

