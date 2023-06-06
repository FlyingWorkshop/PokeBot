import asyncio
import json

from poke_env import ShowdownServerConfiguration, PlayerConfiguration
from poke_env.player import RandomPlayer, Player
from alphagogoat.gogoat import AlphaGogoat

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)



async def main():
    with open("account.json") as f:
        data = json.load(f)


    player_configuration = PlayerConfiguration(data['username'], data['password'])
    # unregistered_player_configuration = PlayerConfiguration

    # We create a random player
    # player = RandomPlayer(
    #     player_configuration=PlayerConfiguration(data['username'], data['password']),
    #     server_configuration=ShowdownServerConfiguration,
    # )
    player = MaxDamagePlayer(
        player_configuration=PlayerConfiguration(data['username'], data['password']),
        server_configuration=ShowdownServerConfiguration,
        avatar='youngn',
        # start_challenging=True,
        # opponent=['flyingworkshop'],
    )

    # Sending challenges to 'your_username'
    # await player.send_challenges('flyingworkshop', n_challenges=1)

    # Accepting one challenge from any user
    await player.accept_challenges(None, 1)
    # player.background_accept_challenge('flyingworkshop')

    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    # await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())