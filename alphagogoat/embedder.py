import json
import logging
import copy

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.effect import Effect
from poke_env.environment.move import Move, DynamaxMove
import torch


from pokedex import make_pokedex


LOGGER = logging.getLogger('poke-env')
GEN = 8
POKEDEX = make_pokedex()
MAX_MOVES = max([len(data['moves']) for data in POKEDEX.values()])
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

    def _embed_move(self, id: str, prob: float, pokemon: Pokemon) -> torch.Tensor:
        """
        >>> embedder = Embedder()
        >>> volcarona = Pokemon(species='Volcarona', gen=8)
        >>> embedder._embed_move("fierydance", 0, volcarona).shape
        """
        move = Move(id, gen=8)
        # TODO: handle moves that require no item like poltergeist
        # TODO: handle moves that consume berries like stuff cheeks
        # TODO: handle knock off
        # TODO: handle curse/hex
        # TODO: handle facade
        embedding = [
            prob,
            move.accuracy,
            move.base_power,
            move.breaks_protect,
            move.category.value,
            move.crit_ratio,
            move.current_pp,  # TODO: how to extract this live from battle?
            move.damage,  # TODO: handle moves like seismic toss
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
            # TODO: potentially handle ghost vs non ghost target
            move.priority,
            move.recoil,
            # TODO: handle secondary effects
            1 if move.self_destruct == 'always' else 0,
            move.self_switch,
            # TODO: handle side conditions
            move.sleep_usable,
            move.steals_boosts,
            # TODO: handle terrain (is this necessary? yes for dynamax moves)
            move.thaws_target,
            move.type.value
            # TODO: handle volatile status (e.g., flinch)
            # TODO: handle weather
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

        return torch.Tensor(embedding)


    def _embed_moves_from_pokemon(self, pokemon: Pokemon):
        """
        >>> embedder = Embedder()
        >>> appletun = Pokemon(gen=8, species="Appletun")
        >>> embedder._embed_moves_from_pokemon(appletun).shape
        torch.Size([8, 33])
        """
        # TODO: handle from most recent data (optional)
        # TODO: handle dynamax moves
        # TODO: handle known moves (IMPORTANT)

        # make move embeddings
        embeddings = []
        moves = POKEDEX[pokemon.species.capitalize()]['moves']
        for name, prob in moves.items():
            id = name.lower().replace(" ", "")
            embedding = self._embed_move(id, prob, pokemon)
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
        >>> abomasnow = Pokemon(gen=8, species="Abomasnow")
        >>> embedder._embed_pokemon(abomasnow).shape
        torch.Size([174])
        """
        # TODO: handle abilities
        # TODO: handle items
        # TODO: handle evs and levels
        stats = pokemon.base_stats
        for stat, val in stats.items():
            if stat == 'hp':
                continue
            stats[stat] = val + 1 * pokemon.boosts[stat]
        stats = [val for stat, val in sorted(stats.items())]

        effects = [0] * len(Effect)
        for effect in pokemon.effects:
            effects[effect.value] = 1

        status = -1 if pokemon.status is None else pokemon.status.value
        status_counter = pokemon.status_counter
        type1 = pokemon.type_1.value
        type2 = pokemon.type_2.value

        embedding = torch.Tensor(stats + effects + [status, status_counter, type1, type2])
        return embedding

    def embed(self, battle: Battle):
        # TODO: maintain a list of team1 and team2 pokemon and what moves they have used
        return


