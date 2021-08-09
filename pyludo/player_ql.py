import random
import logging
import numpy as np
from numpy.lib.arraysetops import isin
from numpy.lib.npyio import save, savetxt

from pyludo.state import ACTION, REWARD, STATE
from pyludo.helpers import randargmax, will_send_opponent_home, will_send_self_home

class LudoPlayerQLearning:
	""" player trained with Q-learning """

	def __init__(self, qtable=None, **kwargs):
		
		# load Q table from file, or initialize empty
		if qtable is None: # empty
			self.qtable = np.zeros((len(STATE), len(ACTION)), dtype=float)
		elif isinstance(qtable, str): # from path
			self.qtable_path = qtable
			self.qtable = np.loadtxt(self.qtable_path, delimiter=",")
		elif isinstance(qtable, np.ndarray): # shared
			self.qtable = qtable
		
		# define options from kwargs (or use default values)
		self.name             = kwargs['name'] if 'name' in kwargs else "q-learning"
		self.training         = kwargs['training'] if 'training' in kwargs else False
		self.advanced         = kwargs['advanced'] if 'advanced' in kwargs else False
		self.decaying_epsilon = kwargs['decaying_epsilon'] if 'decaying_epsilon' in kwargs else False
		self.epsilon          = kwargs['epsilon'] if 'epsilon' in kwargs else 0.05
		self.epsilon_min      = kwargs['epsilon_min'] if 'epsilon_min' in kwargs else 0.01
		self.epsilon_max      = kwargs['epsilon_max'] if 'epsilon_max' in kwargs else self.epsilon
		self.alpha            = kwargs['alpha'] if 'alpha' in kwargs else 0.01
		self.gamma            = kwargs['gamma'] if 'gamma' in kwargs else 0.9
		self.num_max_episodes = kwargs['num_max_episodes'] if 'num_max_episodes' in kwargs else 0.9
		self.cur_episode      = 0
		self.cum_reward       = 0
		
		if self.decaying_epsilon:
			self.set_decaying_epsilon()
		
		# logging.info(self.qtable)
		# logging.info(f'Initialized LudoPlayerQLearning in {"training" if self.training else "playing"} mode.')
		
	def save_qtable(self, path=None):
		np.savetxt("qtable.csv" if path == None else path, self.qtable, delimiter=",", fmt="%f")
		
	def save_info(self, dir):
		fs_info = open(f"{dir}/player_info.txt", "a")
		out = [
		       f"qtable_path: {self.qtable_path if hasattr(self, 'qtable_path') else 'external'}",
		       f"decaying_epsilon: {self.decaying_epsilon}",
		       f"epsilon_min: {self.epsilon_min}",
		       f"epsilon_max: {self.epsilon_max}",
		       f"epsilon: {self.epsilon}",
		       f"alpha: {self.alpha}",
		       f"gamma: {self.gamma}",
		       f"advanced: {self.advanced}",
		       f"training: {self.training}",
		]

		for r in REWARD:
			out.append(repr(r))

		fs_info.write("\n".join(out))
		
	def set_decaying_epsilon(self):
		r = max((self.num_max_episodes - self.cur_episode)/self.num_max_episodes, 0)
		self.epsilon = (self.epsilon_max - self.epsilon_min) * r + self.epsilon_min
		# logging.info(f"changed epsilon to {self.epsilon}")
		
	def new_episode(self):
	
		if self.num_max_episodes is None:
			return
			
		self.cur_episode += 1
		self.cum_reward = 0
		
		if self.decaying_epsilon:
			self.set_decaying_epsilon()

	def get_action(self, state, next_states_actions, epsilon=0.05):
		
		# get a list of tuples of current (STATE(), ACTION())
		if self.advanced:
			states_actions = np.array([(state.get_state_advanced(i), next_states_actions[i,1]) for i in range(4)], dtype=np.object_)
		else:
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
		reward_name, reward = state.get_reward(next_state, bonus=self.advanced)
		# logging.info(f"q reward: {reward_name} ({reward})")
		self.cum_reward += reward
		
		# state indices
		if self.advanced:
			state_idx = state.get_state_advanced(token_id)
			next_state_idx = next_state.get_state_advanced(token_id)
		else:
			state_idx = state.get_state(token_id)
			next_state_idx = next_state.get_state(token_id)

		# compute Q value of most optimal action for next state (s')
		# q_next = max(self.qtable[next_state.get_state(token_id), :])
		q_next = max(self.qtable[next_state_idx, :])
		# logging.info(f"q_next {q_next}")
		
		# compute new Q value
		# q_current = self.qtable[state.get_state(token_id), action]
		q_current = self.qtable[state_idx, action]
		q_new = q_current + self.alpha * (reward + self.gamma * q_next - q_current)
		# logging.info(f"q_current {q_current}, q_next: {q_next}")
		
		# update Q table
		# self.qtable[state.get_state(token_id), action] = q_new
		self.qtable[state_idx, action] = q_new

		return token_id