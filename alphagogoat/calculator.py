from poke_env.environment import Pokemon, Move, MoveCategory, Battle, Weather, PokemonType, Status
import random
from math import floor

from .pokedex import POKEDEX
from .constants import DEFAULT_EVS, DEFAULT_IVS, EVS_PER_INC

def average_pokemon_stats():
    overall_stats = {}
    ev = DEFAULT_EVS
    defaults = {'hp': 80, 'atk': 90, 'def': 83, 'spa': 83, 'spd': 83, 'spe': 78}

    for stat, base in defaults.items():
        overall_stats[stat] = floor(
            (2 * base + DEFAULT_IVS + ev // 4) * 100 / 100
        ) + 5

    return overall_stats

def calc_stats(pokemon: Pokemon):
    # TODO: handle dynamax
    overall_stats = {}
    data = POKEDEX[pokemon.species]
    for stat, base in pokemon.base_stats.items():
        ev = DEFAULT_EVS
        if "evs" in data and stat in data["evs"]:
            ev += data["evs"][stat]
        if stat == 'hp' and pokemon.species != 'shedinja':
            overall_stats[stat] = floor(
                (2 * base + DEFAULT_IVS + ev // 4) * pokemon.level / 100
            ) + pokemon.level + 10
        else:
            overall_stats[stat] = floor(
                (2 * base + DEFAULT_IVS + ev // 4) * pokemon.level / 100
            ) + 5

    # apply boosts
    for stat, boost in pokemon.boosts.items():
        if stat in overall_stats and boost:
            overall_stats[stat] *= boost

    return overall_stats

def calc_damage(attacking: Pokemon, move: Move, target: Pokemon, turn: Battle, is_critical=False, unknownOpp = False):
    attacking_stats = calc_stats(attacking)
    if unknownOpp:
        target_stats = average_pokemon_stats()
    else:
        target_stats = calc_stats(target)
    level = attacking.level
    if move.category == MoveCategory.PHYSICAL:
        A = attacking_stats['atk']
    elif move.category == MoveCategory.SPECIAL:
        A = attacking_stats['spa']
    else:  # move.category == MoveCategory.STATUS
        return (0, 0)

    if move.category == MoveCategory.PHYSICAL:
        D = target_stats['def']
    elif move.category == MoveCategory.SPECIAL:
        D = target_stats['spd']
    else:  # move.category == MoveCategory.STATUS
        return (0, 0)

    power = move.base_power
    weather = 1
    if turn.weather == Weather.RAINDANCE:
        if move.type == PokemonType.WATER:
            weather = 1.5
        if move.type == PokemonType.FIRE:
            weather = 0.5
    if turn.weather == Weather.SUNNYDAY:
        if move.type == PokemonType.FIRE:
            weather = 1.5
        if move.type == PokemonType.WATER:
            weather = 0.5

    if move.id in ['wickedblow', 'stormthrow', 'frostbreath', 'zippyzap', 'surgingstrikes']:
        is_critical = True
    if not unknownOpp and target.ability in ['battlearmor', 'shellarmor']:
        is_critical = False
    critical = 1.5 if is_critical else 1

    STAB = 1
    if attacking.type_1 == move.type or attacking.type_2 == move.type or attacking.ability == 'adaptability':
        STAB = 1.5

    if unknownOpp:
        type_effectiveness = 1
    else:
        type_effectiveness = target.damage_multiplier(move)
        if move.id == 'thousandarrows' and target.type_1 == PokemonType.FLYING or target.type_2 == PokemonType.FLYING:
            type_effectiveness = 1
        # TODO: handle if target is grounded
        if attacking.ability == 'scrappy':
            is_ghost = target.type_1 == PokemonType.GHOST or target.type_2 == PokemonType.GHOST
            if is_ghost and (move.type == PokemonType.NORMAL or move.type == PokemonType.FIGHTING):
                type_effectiveness = 1
        if move.id == 'freezedry' and (target.type_1 == PokemonType.WATER or target.type_2 == PokemonType.WATER):
            type_effectiveness = 2

    # TODO: figure out if pokemon showdown bundles this into boost
    burn = 1
    if move.id != 'facade' and attacking.status == Status.BRN and move.category == MoveCategory.PHYSICAL:
        burn = 0.5

    other = 1
    # TODO: balloon, choice items, assault vest, life orb, luster orb, light ball, eviolalite

    damage = (
        (((2 * level) / 5 + 2) * power * (A / D)) / 50 + 2
    ) * weather * critical * STAB * type_effectiveness * burn * other
    min_damage = floor(damage * 0.85)
    max_damage = floor(damage)
    return (min_damage, max_damage)
