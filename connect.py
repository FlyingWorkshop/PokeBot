import asyncio
import json

from poke_env import ShowdownServerConfiguration, PlayerConfiguration
from poke_env.player import RandomPlayer


async def main():
    with open("account.json") as f:
        data = json.load(f)


    # We create a random player
    player = RandomPlayer(
        player_configuration=PlayerConfiguration(data['username'], data['password']),
        server_configuration=ShowdownServerConfiguration,
    )

    # Sending challenges to 'your_username'
    # await player.send_challenges('flyingworkshop', n_challenges=1)

    # Accepting one challenge from any user
    await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    # await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    # for battle in player.battles.values():
    #     print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())