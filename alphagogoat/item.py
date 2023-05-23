from pokedex import POKEDEX
from enum import IntEnum


def _make_enum():
    item_catalog = set()
    for pokemon, data in POKEDEX.items():
        item_catalog.update(data['items'])

    item_catalog = sorted(list(item_catalog))
    Item = IntEnum('Item', item_catalog)
    return Item


Item = _make_enum()
