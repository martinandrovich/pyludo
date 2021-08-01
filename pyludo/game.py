import random
import logging
from datetime import datetime
import numpy as np

from pyludo.players import PLAYER_COLORS
from pyludo.helpers import star_jump, is_globe_pos
from pyludo.state_space import ACTION

class LudoState:
	def __init__(self, state=None, empty=False):
		
		random.seed(datetime.now())
		
		if state is not None:
			self.state = state
		else:
			self.state = np.empty((4, 4), dtype=np.int)  # 4 players, 4 tokens per player
			if not empty:
				self.state.fill(-1)

	def copy(self):
		return LudoState(self.state.copy())

	def __getitem__(self, item):
		return self.state[item]

	def __setitem__(self, key, value):
		self.state[key] = value

	def __iter__(self):
		return self.state.__iter__()

	@staticmethod
	def get_tokens_relative_to_player(tokens, player_id):
		if player_id == 0:
			return tokens

		rel_tokens = []
		for token_id, token_pos in enumerate(tokens):
			if token_pos == -1 or token_pos == 99:  # start and end pos are independent of player id
				rel_tokens.append(token_pos)
			elif token_pos < 52:  # in common area
				rel_tokens.append((token_pos - player_id * 13) % 52)
			else:  # in end area, 52 <= x < 52 + 20
				rel_tokens.append(((token_pos - 52 - player_id * 5) % 20) + 52)
		return rel_tokens

	def get_state_relative_to_player(self, rel_player_id, keep_player_order=False):
		if rel_player_id == 0:
			return self.copy()

		rel = LudoState(empty=True)
		new_player_ids = list(range(4)) if keep_player_order else [(x - rel_player_id) % 4 for x in range(4)]
		
		for player_id, player_tokens in enumerate(self):
			new_player_id = new_player_ids[player_id]
			rel[new_player_id] = self.get_tokens_relative_to_player(player_tokens, rel_player_id)

		return rel

	def move_token(self, token_id, dice_roll, is_jump=False):
		""" move token for player 0 """
		
		cur_pos = self[0][token_id]
		
		# if token in goal, no actions possible
		if cur_pos == 99:
			return False, ACTION.NONE

		new_state = self.copy()
		player = new_state[0]
		opponents = new_state[1:]

		# move from home if token in home and a 6 is rolled
		if cur_pos == -1:
			if dice_roll != 6:
				return False, ACTION.NONE
				
			player[token_id] = 1
			opponents[opponents == 1] = -1
			
			return new_state, ACTION.MOVE_FROM_HOME

		target_pos = cur_pos + dice_roll

		# common area move
		if target_pos < 52:
		
			occupants = opponents == target_pos
			occupant_count = np.sum(occupants)
			
			# occupied by multiple other tokens
			if occupant_count > 1:
				player[token_id] = -1  # sends self home
				return new_state, ACTION.MOVE_ONTO_ANOTHER_DIE
			
			# globe
			if (occupant_count == 1 and is_globe_pos(target_pos)):
				player[token_id] = -1  # sends self home
				return new_state, ACTION.MOVE_ONTO_GLOBE_WHERE_OTHER
			elif (is_globe_pos(target_pos)):
				player[token_id] = target_pos
				return new_state, ACTION.MOVE_ONTO_GLOBE
			
			# star
			if (star_jump_length := star_jump(target_pos)):
				
				kill = False
				if occupant_count == 1:
					opponents[occupants] = -1
					kill = True
					
				if target_pos == 51:
					player[token_id] = 99  # send directly to goal
					return new_state, ACTION.MOVE_ONTO_GOAL
					
				else:
					player[token_id] = target_pos + star_jump_length
					return new_state, (ACTION.MOVE_ONTO_STAR_AND_KILL if kill else ACTION.MOVE_ONTO_STAR)
					
			# normal move
			player[token_id] = target_pos
			
			if occupant_count == 1:
				opponents[occupants] = -1
				return new_state, ACTION.MOVE_ONTO_ANOTHER_KILL
			else:
				return new_state, ACTION.MOVE

		# victory road move
		if target_pos == 57:  # token reached goal
			player[token_id] = 99
			return new_state, ACTION.MOVE_ONTO_GOAL
			
		elif target_pos < 57:  # no goal bounce
			player[token_id] = target_pos
			return new_state, ACTION.MOVE_ONTO_VICTORY_ROAD
			
		else:  # bounce back from goal pos
			player[token_id] = 57 - (target_pos - 57)
		return new_state, ACTION.MOVE_ONTO_VICTORY_ROAD

	def get_winner(self):
		for player_id in range(4):
			if np.all(self[player_id] == 99):
				return player_id
		return -1


class LudoStateFull:
	def __init__(self, state, roll, next_states):
		self.state = state
		self.roll = roll
		self.next_states = next_states


class LudoGame:
	def __init__(self, players, state=None, info=False):
		assert len(players) == 4, "There must be four players in the game."
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
		dice_roll = random.randint(1, 6)
		logging.info("Dice rolled a {} for player {} [{}].".format(dice_roll, PLAYER_COLORS[self.currentPlayerId], player.name))
		
		# create relative state for
		relative_state = self.state.get_state_relative_to_player(self.currentPlayerId)

		# get an array possible state-action pairs for each token
		# each entry corresponds to a token and its the next state moving that token
		state_action_pairs = np.array(
			[relative_state.move_token(token_id, dice_roll) for token_id in range(4)]
		)
		
		# extract list of next possible states (tuple unpacking)
		# https://stackoverflow.com/questions/12142133/how-to-get-first-element-in-a-list-of-tuples
		rel_next_states = np.array([i[0] for i in state_action_pairs])

		# if there are possible moves, call the .play() method for the player
		if np.any(rel_next_states != False):
			
			# make a move; return index of the token that is wished to be moved
			token_id = player.play(relative_state, dice_roll, rel_next_states)

			# change from [n] to n (remove array)
			if isinstance(token_id, np.ndarray):
				token_id = token_id[0]

			if rel_next_states[token_id] is False:
				logging.warning("Player has chosen an invalid move. Choosing first valid move.")
				token_id = np.argwhere(rel_next_states != False)[0][0]
			
			# update state with chosen action
			state_prev = self.state
			self.state = rel_next_states[token_id].get_state_relative_to_player((-self.currentPlayerId) % 4)
			logging.info("Moved token {} of player {} [{}] from {} to {}.".format(token_id, PLAYER_COLORS[self.currentPlayerId], player.name, state_prev[self.currentPlayerId][token_id], self.state[self.currentPlayerId][token_id]))
			logging.info("Action: {}".format(state_action_pairs[token_id, 1]))

	def play_full_game(self):
		while self.state.get_winner() == -1:
			self.step()
		return self.state.get_winner()
