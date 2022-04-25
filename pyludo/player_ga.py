import numpy as np
from pyludo.state import ACTION, REWARD, STATE

class LudoPlayerGA:
	""" player trained with Genetic Algorithm via pytorch """

	def __init__(self, weights=None, **kwargs):

		# load model
		self.name  = kwargs['name'] if 'name' in kwargs else "genetic-algorithm"
		self.weights = weights
		# self.weights = np.random.rand(len(ACTION))

	def play(self, state, dice_roll, next_states_actions):

		# map playable actions to corresponding weight
		next_actions = next_states_actions[:, 1]
		weighted_actions = [self.weights[action] if action != ACTION.NONE else -np.inf for action in next_actions]

		# select token with best action
		token_id = np.argmax(weighted_actions)

		return token_id