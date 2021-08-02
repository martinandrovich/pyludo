import random
import numpy as np

from pyludo.helpers import token_vulnerability

"""
def play(self, state, dice_roll, next_states):
	:param state:
		current state relative to this player
	:param dice_roll:
		[1, 6]
	:param next_states:
		np array of length 4 with each entry being the next state moving the corresponding token.
		False indicates an invalid move. 'play' won't be called, if there are no valid moves.
	:return:
		index of the token that is wished to be moved. If it is invalid, the first valid token will be chosen.
"""

PLAYER_COLORS = ['green', 'blue', 'red', 'yellow']

class LudoPlayerRandom:
	""" takes a random valid action """
	name = 'random'

	@staticmethod
	def play(state, dice_roll, next_states_actions):

		next_states = np.array([i[0] for i in next_states_actions])
		return random.choice(np.argwhere(next_states != False))

class LudoPlayerFast:
	""" moves the furthest token that can be moved """
	name = 'fast'

	@staticmethod
	def play(state, _, next_states_actions):

		next_states = np.array([i[0] for i in next_states_actions])

		for token_id in np.argsort(state[0]):
			if next_states[token_id] is not False:
				return token_id

class LudoPlayerAggressive:
	""" tries to send the opponent home, else random valid move """
	name = 'aggressive'

	@staticmethod
	def play(state, dice_roll, next_states_actions):

		next_states = np.array([i[0] for i in next_states_actions])

		for token_id, next_state in enumerate(next_states):
			if next_state is False:
				continue
			if np.sum(next_state[1:] == -1) > np.sum(state[1:] == -1):
				return token_id

		return LudoPlayerRandom.play(None, None, next_states_actions)

class LudoPlayerDefensive:
	""" moves the token that can be hit by most opponents """
	name = 'defensive'

	@staticmethod
	def play(state, dice_roll, next_states_actions):

		next_states = np.array([i[0] for i in next_states_actions])

		hit_rates = np.empty(4)
		hit_rates.fill(-1)

		for token_id, next_state in enumerate(next_states):
			if next_state is False:
				continue
			hit_rates[token_id] = token_vulnerability(state, token_id)
		return random.choice(np.argwhere(hit_rates == np.max(hit_rates)))
