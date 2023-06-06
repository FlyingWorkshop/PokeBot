from alphagogoat.utils import make_data
from alphagogoat.calculator import calc_damage
from poke_env.environment import Pokemon, Move, Battle

def main():
    data = make_data(f"cache/replays/gen8randombattle-1197381786.json")
    turns, actions1, actions2 = data

    for turn, action1, action2 in zip(turns, actions1, actions2):
        print(turn.turn)
        print(f"\t{turn.active_pokemon.species} [{turn.active_pokemon.current_hp}]")
        print(f"\t{action1}")
        damage = (0, 0)
        if action1[0] == 'move':
            move = Move(action1[1], 8)
            damage = calc_damage(turn.active_pokemon, move, turn.opponent_active_pokemon, turn)
        print(f"\t{damage}")
        print(f"\t{turn.opponent_active_pokemon.species} [{turn.opponent_active_pokemon.current_hp}]")
        print(f"\t{action2}")
        if action2[0] == 'move':
            move = Move(action2[1], 8)
            damage = calc_damage(turn.opponent_active_pokemon, move, turn.active_pokemon, turn)
        print(f"\t{damage}")


if __name__ == "__main__":
    main()