import random
import numpy as np

from pyludo.state import ACTION, STATE
from pyludo.helpers import randargmax

class LudoPlayerQLearning:
	""" player trained with Q-learning """
	name = 'q-learning'

	def __init__(self, training=False):

		self.qtable = np.zeros((len(STATE), len(ACTION)))
		self.training = training
		
		self.qtable = np.array([
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			[0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
		])
		
		print(self.qtable)
		print(f'Initialized LudoPlayerQLearning in {"training" if self.training else "playing"} mode.')
	

	@staticmethod
	def observe_reward_and_state(state, next_states_actions, action):
		return 0.5, "next state"

	def get_action(self, policy, state, next_states_actions, **kwargs):
		assert isinstance(policy, str) and (policy in ["optimal", "e-greedy"]), "Wrong policy specified."
		
		# get a list of tuples of current (STATE(), ACTION())
		states_actions = np.array([(state.get_state(i), next_states_actions[i,1]) for i in range(4)], dtype=np.object_)
		# print(np.array([(state.get_state(i), next_states_actions[i,1]) for i in range(4)], dtype=np.object_))
		print(states_actions)
		
		# create an array of Q values for the given state-actions pairs
		q_values = np.array(
			[self.qtable[tuple(states_actions[i])] for i in range(4)]
		)
		print("q_values: ", q_values)
		
		# plausible action(s) from Q table
		# a = (q_values != None) & (states_actions[:, 1] != ACTION.NONE)
		# print(f"a: {a}")
		
		# set epsilon (chance for exploration/random action)
		epsilon = 0 if policy == "optimal" else kwargs["epsilon"] if "epsilon" in kwargs else 0.05
		# print("epsilon: ", epsilon)
		
		p = random.random()
		# print(p)
		
		if p < epsilon: # exploration
			print("exploration")
			# select random plausible action
			return random.choice(np.argwhere((q_values != None) & (states_actions[:, 1] != ACTION.NONE)))
			# return random.choice(np.argwhere(states_actions[:,1] != ACTION.NONE))
			
		else: # exploitation
			print("exploitation")
			# get argmax (index) of plausible Q-value
			# i = np.argmax(np.where((q_values != None) & (states_actions[:, 1] != ACTION.NONE), q_values, np.NINF))
			# i = np.argmax(np.random.random(q_values.shape) * (np.where((q_values != None) & (states_actions[:, 1] != ACTION.NONE), q_values, np.NINF)))
			i = randargmax(np.where((q_values != None) & (states_actions[:, 1] != ACTION.NONE), q_values, np.NINF))
			print(f"max Q value of {q_values[i]} at index {i}")
			return i

	def play(self, state, dice_roll, next_states_actions):

		# chose action a using policy derived from Q
		# if training, action is epsilon-greedy with varying epsilon
		
		if (self.training):
			action = self.train(state, next_states_actions)
		else:
			action = self.get_action("optimal", state, next_states_actions)
			
		return action

	def train(self, state, next_states_actions):

		action = self.get_action("e-greedy", state, next_states_actions, epsilon=0.1)

		# take action a, observe r and s'
		reward, state_next = self.observe_reward_and_state(state, next_states_actions, action)

		return action