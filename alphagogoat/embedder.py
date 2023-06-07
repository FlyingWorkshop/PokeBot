import copy
import json
import logging
import re
import random

import torch
from poke_env.environment import Battle, Effect, Move, Pokemon, Status, SideCondition, Weather, Field
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

from .calculator import calc_stats
from .constants import MAX_MOVES, MAX_ABILITIES, MAX_ITEMS, BOOSTABLE_STATS, DEFAULT_EVS, DEFAULT_IVS, EVS_PER_INC, DEVICE


class Embedder:
    def __init__(self):
        pass

    @staticmethod
    def embed_conditions(battle: Battle, opponent_pov: bool) -> torch.FloatTensor:
        # takes in a battle object, and returns a tensor filled with the
        # side conditions of both sides, field, and weather of both sides
        d1 = battle.opponent_side_conditions if opponent_pov else battle.side_conditions
        d2 = battle.side_conditions if opponent_pov else battle.opponent_side_conditions

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
        for f in Field:
            if f in battle.fields and battle.fields[f]:
                embed.append(battle.fields[f])
            else:
                embed.append(0)

        return torch.FloatTensor(embed)

    @staticmethod
    def embed_pokemon(pokemon: Pokemon, is_active: bool) -> torch.Tensor:
        data = POKEDEX[pokemon.species]
        embedding = [
            is_active,
            pokemon.current_hp or 0,  # handles when current_hp is None
            pokemon.first_turn,
            pokemon.is_dynamaxed,
            pokemon.level,
            pokemon.must_recharge or pokemon.preparing,  # TODO: handle recharge and preparing
            pokemon.protect_counter,
            pokemon.type_1.value,
            -1 if pokemon.type_2 is None else pokemon.type_2.value,
            -1 if pokemon.status is None else pokemon.status.value,
            pokemon.status_counter,
            pokemon.weight,
        ]

        # abilities
        abilities = []
        if pokemon.ability == 'unknown_ability' or pokemon.ability is None:
            for ability, prob in data['abilities'].items():
                abilities += [prob, Ability[ability].value]
        else:
            abilities += [1, Ability[pokemon.ability].value]
        abilities += [0] * (2 * MAX_ABILITIES - len(abilities))

        # items
        items = []
        if pokemon.item == 'unknown_item' or pokemon.item is None:
            for item, prob in data['items'].items():
                items += [prob, Item[item].value]
        else:
            items += [1, Item[pokemon.item].value]
        items += [0] * (2 * MAX_ITEMS - len(items))

        # effects
        effects = [0] * len(Effect)
        for effect in pokemon.effects:
            effects[effect.value - 1] = 1
        # effects = [
        #     Effect['CONFUSION'] in pokemon.effects,
        #     Effect['ENCORE'] in pokemon.effects,
        #     Effect['FLASH_FIRE'] in pokemon.effects,
        #     any(e in pokemon.effects for e in (Effect['FIRE_SPIN'], Effect['TRAPPED'], Effect['MAGMA_STORM'], Effect['WHIRLPOOL'])),
        #     Effect['LEECH_SEED'] in pokemon.effects,
        #     Effect['STICKY_WEB'] in pokemon.effects,
        #     Effect['SUBSTITUTE'] in pokemon.effects,
        #     Effect['YAWN'] in pokemon.effects,
        #     Effect['NO_RETREAT'] in pokemon.effects,
        #     Effect['MAGNET_RISE'] in pokemon.effects
        # ]

        # stats
        stats = [value for _, value in sorted(calc_stats(pokemon).items())]
        embedding = torch.Tensor(abilities + items + stats + effects + embedding)
        return embedding

