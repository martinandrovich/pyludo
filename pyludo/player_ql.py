import random
import numpy as np

class LudoPlayerQLearning:
	""" player trained with Q-learning """
	name = 'q-learning'

	def __init__(self, train=False):
		self.qtable = 0
		self.train = train

	@staticmethod
	def observe_reward_and_state(state, next_states, action):
		return 0.5, "next state"

	@staticmethod
	def get_action(type, next_states, **options):
		assert isinstance(type, str) and (type in ["optimal", "e-greedy"]), "Wrong function parameters!"
		
		return random.choice(np.argwhere(next_states != False))

	def play(self, state, dice_roll, next_states):

		# chose action a using policy derived from Q
		# if training, action is epsilon-greedy with varying epsilon
	
		if (self.train):
			action = self.train(state, next_states)
		else:
			action = self.get_action("optimal", next_states)

		# return the desired action to the game
		return action

	def train(self, state, next_states):
	
		action = self.get_action("e-greedy", next_states, epsilon=0.9)
		
		# take action a, observe r and s'
		reward, state_next = self.observe_reward_and_state(state, next_states, action)
		
		return action