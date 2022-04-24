from lib2to3.pgen2 import token
import numpy as np
import pygad
import pygad.torchga
import torch
from sklearn.preprocessing import OneHotEncoder
from pyludo.state import ACTION, REWARD, STATE
from pyludo.helpers import randargmax, will_send_opponent_home, will_send_self_home

class LudoPlayerGA:
	""" player trained with Genetic Algorithm via pytorch """

	def __init__(self, model, **kwargs):

		# load model
		self.name  = kwargs['name'] if 'name' in kwargs else "genetic-algorithm"
		self.model = model

	def play(self, state, dice_roll, next_states_actions):

		# state = LudoState()
		# state.get_state(i) returns token STATE (enum) for the i'th token (piece)
		# next_states_actions = [(state, ACTION), (state, ACTION), ...] or [False, False, (state, ACTION), False], etc.
		
		# current state, advanced, onehot (56)
		states_oh = np.hstack([state.get_state_onehot(i, advanced=True) for i in range(4)]).reshape([1, 56])

		# next states (16)
		# states_oh = np.hstack([[state.get_state_onehot(i, advanced=True) if state else np.zeros((1, len(STATE))) for i in range(4)] for state in next_states_actions[:,0]]).reshape([1, 224])

		predictions = self.model(torch.tensor(states_oh, dtype=torch.float))
		# predictions = pygad.torchga.predict(model=self.model, solution=self.solution, data=torch.tensor(states_oh, dtype=torch.float))

		token_id = int(torch.argmax(predictions))
		return token_id