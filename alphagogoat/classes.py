import poke_env

class MaxDamagePlayer(poke_env.player.Player):
    def __init__(self):
        super().__init__()

    def calc_value(self, battle):
        enemy_team = None

    def monte_carlo_tree_search(self, battle):
        pass

    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

class EnemyPokemon(poke_env.environment.pokemon.Pokemon):
    def __init__(self, gen: int):
        super().__init__(gen)