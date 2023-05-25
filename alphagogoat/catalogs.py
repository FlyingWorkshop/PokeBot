from pokedex import POKEDEX
from enum import IntEnum
from poke_env.environment.move import Move
from poke_env.environment.side_condition import SideCondition

import re


def _make_item_enum():
    item_catalog = set()
    for pokemon, data in POKEDEX.items():
        item_catalog.update(data['items'])

    item_catalog = sorted(list(item_catalog))
    Item = IntEnum('Item', item_catalog)
    return Item


Item = _make_item_enum()


def _make_volatile_status_enum():
    move_catalog = set()
    for pokemon, data in POKEDEX.items():
        move_catalog.update(data['moves'])

    move_catalog = [Move(re.sub("\s|-|'", "", move.lower()), 8) for move in sorted(list(move_catalog))]
    volatile_status_catalog = set()
    for move in move_catalog:
        if move.volatile_status is None:
            continue
        volatile_status_catalog.add(move.volatile_status)

    VolatileStatus = IntEnum("VolatileStatus", list(sorted(volatile_status_catalog)))
    return VolatileStatus


VolatileStatus = _make_volatile_status_enum()


def _make_side_condition_mapping():
    return {elem.name.lower().replace("_", ""): elem.name for elem in SideCondition}


SIDE_COND_MAP = _make_side_condition_mapping()


def _make_move_enum():
    move_enum = set()
    for pokemon, data in POKEDEX.items():
        for move in data['moves']:
            move_enum.add(re.sub(r"\s|-|'", "", move.lower()))
    
    move_enum.add('struggle')
    move_enum.add('spiritshackle')
    #move_enum.add('fishiousrend')
    MoveEnum = IntEnum("MoveEnum", list(sorted(move_enum)))
    return MoveEnum


MoveEnum = _make_move_enum()

