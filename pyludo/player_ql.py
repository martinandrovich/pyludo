import random
import logging
import numpy as np
from numpy.lib.arraysetops import isin
from numpy.lib.npyio import save, savetxt

from pyludo.state import ACTION, REWARD, STATE
from pyludo.helpers import randargmax, will_send_opponent_home, will_send_self_home

class LudoPlayerQLearning:
	""" player trained with Q-learning """
	name = 'q-learning'

	def __init__(self, training=False, qtable=None, decaying_epsilon=False, num_max_episodes=None):
		
		# load Q table from file, or initialize empty
		
		if qtable is None: # empty
			self.qtable = np.zeros((len(STATE), len(ACTION)), dtype=float)
		elif isinstance(qtable, str): # from path
			self.qtable = np.loadtxt(qtable, delimiter=",")
		elif isinstance(qtable, np.ndarray): # shared
			self.qtable = qtable
		# try:
		# 	self.qtable = np.zeros((len(STATE), len(ACTION)), dtype=float) if qtable == None else np.loadtxt(qtable, delimiter=",")
		# except:
		# 	logging.error("Could not load Q-table, initialized an empty table.")
		# 	self.qtable = np.zeros((len(STATE), len(ACTION)), dtype=float)
		
		self.training = training
		self.epsilon = 0.05 # e-greedy
		self.decaying_epsilon = decaying_epsilon
		self.epsilon_min = 0.01
		self.epsilon_max = 0.7
		self.alpha = 0.001 # learning rate # 0.0001
		self.gamma = 0.9 # discount factor
		self.cur_episode = 0
		self.num_max_episodes = num_max_episodes
		
		if self.decaying_epsilon:
			self.set_decaying_epsilon()
		
		# logging.info(self.qtable)
		# logging.info(f'Initialized LudoPlayerQLearning in {"training" if self.training else "playing"} mode.')
		
	def save_qtable(self, path=None):
		np.savetxt("qtable.csv" if path == None else path, self.qtable, delimiter=",", fmt="%f")
		
	def save_info(self, dir):
		fs_info = open(f"{dir}/info.txt", "a")
		out = [f"decaying_epsilon: {self.decaying_epsilon}",
		       f"epsilon_min: {self.epsilon_min}",
		       f"epsilon_max: {self.epsilon_max}",
		       f"epsilon: {self.epsilon}",
		       f"alpha: {self.alpha}",
		       f"gamma: {self.gamma}"]

		fs_info.write("\n".join(out))
		
	def set_decaying_epsilon(self):
		r = max((self.num_max_episodes - self.cur_episode)/self.num_max_episodes, 0)
		self.epsilon = (self.epsilon_max - self.epsilon_min) * r + self.epsilon_min
		# logging.info(f"changed epsilon to {self.epsilon}")
		
	def new_episode(self):
	
		if self.num_max_episodes is None:
			return
			
		self.cur_episode += 1
		
		if self.decaying_epsilon:
			self.set_decaying_epsilon()

	def get_action(self, state, next_states_actions, epsilon=0.05):
		
		# get a list of tuples of current (STATE(), ACTION())
		states_actions = np.array([(state.get_state(i), next_states_actions[i,1]) for i in range(4)], dtype=np.object_)
		# logging.info(states_actions)
		
		# create an array of Q values for the given state-actions pairs
		# use [state action] tuple to index in qtable
		q_values = np.array(
			[self.qtable[tuple(states_actions[i])] for i in range(4)]
		)
		# logging.info(f"q_values:  {q_values}")
		
		# plausible action(s) from Q table
		# a = (q_values != None) & (states_actions[:, 1] != ACTION.NONE)
		# logging.info(f"plausible Q values: {a}")
		
		# epsilon greedy; epsilon is decaying if self.decaying_epsilon=True
		p = random.random()
		# logging.info(f"p: {p}, epsilon: {epsilon}")
		
		if p < epsilon: # exploration
			# logging.info("Exploration")
			# select random plausible action
			# return random.choice(np.argwhere(states_actions[:,1] != ACTION.NONE))
			token_id = random.choice(np.argwhere((q_values != None) & (states_actions[:, 1] != ACTION.NONE)))
			
		else: # exploitation
			# logging.info("Exploitation")
			# get argmax (index) of plausible Q-value
			token_id = randargmax(np.where((q_values != None) & (states_actions[:, 1] != ACTION.NONE), q_values, np.NINF))
			# logging.info(f"max Q value of {q_values[token_id]} at index {token_id}")
		
		# change from [n] to n (remove array)
		if isinstance(token_id, np.ndarray):
			token_id = token_id[0]
			
		return token_id

	def play(self, state, dice_roll, next_states_actions):

		# chose action a using policy derived from Q
		# if training, action is epsilon-greedy with varying epsilon
		
		if (self.training):
			action = self.train(state, next_states_actions)
		else:
			action = self.get_action(state, next_states_actions, epsilon=0) # optimal action
			
		return action

	def train(self, state, next_states_actions):
		
		# take action a (index of token to be played)
		token_id = self.get_action(state, next_states_actions, epsilon=self.epsilon)
		# logging.info(f"token_id: {token_id}")
		action = next_states_actions[token_id, 1]
		# logging.info(f"q action: {action.name}")
		
		# observe r and s'
		next_state = next_states_actions[token_id, 0]
		reward = state.get_reward(next_state)
		# logging.info(f"q reward: {reward}")
		reward = reward.value if isinstance(reward, REWARD) else reward

		# compute Q value of most optimal action for next state (s')
		q_next = max(self.qtable[next_state.get_state(token_id), :])
		# logging.info(f"q_next {q_next}")
		
		# compute new Q value
		q_current = self.qtable[state.get_state(token_id), action]
		q_new = q_current + self.alpha * (reward + self.gamma * q_next - q_current)
		# logging.info(f"q_current {q_current}, q_next: {q_next}")
		
		# update Q table
		self.qtable[state.get_state(token_id), action] = q_new

		return token_id