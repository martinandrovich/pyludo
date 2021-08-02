import random
import logging
from datetime import datetime
import numpy as np

from pyludo.players import PLAYER_COLORS
from pyludo.helpers import star_jump, is_globe_pos
from pyludo.state import LudoState, LudoStateFull

class LudoGame:

	def __init__(self, players, state=None, info=False):

		assert len(players) == 4, "There must be four players in the game."

		random.seed(datetime.now())

		self.players = players
		self.currentPlayerId = -1
		self.state = LudoState() if state is None else state

		if info:
			logging.basicConfig(level=logging.INFO)
		else:
			logging.basicConfig(level=logging.WARNING)

	def step(self):

		# advance player
		self.currentPlayerId = (self.currentPlayerId + 1) % 4
		player = self.players[self.currentPlayerId]

		# roll dice
		# dice_roll = random.randint(1, 6)
		dice_roll = 6
		logging.info("Dice rolled a {} for player {} [{}].".format(dice_roll, PLAYER_COLORS[self.currentPlayerId], player.name))
		
		# create relative state to current player
		relative_state = self.state.get_state_relative_to_player(self.currentPlayerId)

		# get an array possible state-action pairs for each token
		# each entry corresponds to a tuple of (new State(), ACTION() taken) for a given token
		next_states_actions = np.array(
			[relative_state.move_token(token_id, dice_roll) for token_id in range(4)]
		)

		# extract list of next possible states (tuple unpacking)
		# https://stackoverflow.com/questions/12142133/how-to-get-first-element-in-a-list-of-tuples
		# rel_next_states = np.array([i[0] for i in next_states_actions])

		# if there are possible moves, call the .play() method for the player
		# if np.any(rel_next_states != False):
		if np.any(next_states_actions[:,0] != False):

			# make a move; return index of the token that is wished to be moved
			token_id = player.play(relative_state, dice_roll, next_states_actions)
			# print("got token id: ", token_id)

			# change from [n] to n (remove array)
			if isinstance(token_id, np.ndarray):
				token_id = token_id[0]

			# check for invalid moves
			if next_states_actions[token_id,0] is False:
				logging.warning("Player has chosen an invalid move. Choosing first valid move.")
				token_id = np.argwhere(next_states_actions[:,0] != False)[0][0]

			# update state with chosen action
			state_prev = self.state
			self.state = next_states_actions[token_id,0].get_state_relative_to_player((-self.currentPlayerId) % 4)
			logging.info("Moved token {} of player {} [{}] from {} to {}.".format(token_id, PLAYER_COLORS[self.currentPlayerId], player.name, state_prev[self.currentPlayerId][token_id], self.state[self.currentPlayerId][token_id]))
			logging.info("Action: {}".format(next_states_actions[token_id, 1].name))


	def play_full_game(self):
		while self.state.get_winner() == -1:
			self.step()
		return self.state.get_winner()
