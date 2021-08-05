import numpy as np
from aenum import Enum, IntEnum, NoAlias

from pyludo.helpers import star_jump, is_globe_pos, steps_taken, will_send_self_home, will_send_opponent_home, will_send_self_onto_goal, will_send_self_onto_victory_road, will_win_game, will_move_from_home, steps_taken

# Using Enum item as a list index
# https://stackoverflow.com/questions/56650979/using-enum-item-as-a-list-index

# Using Enum with duplicate values (for rewards)
# https://stackoverflow.com/questions/31537316/python-enums-with-duplicate-values

class ACTION(IntEnum):

	MOVE_FROM_HOME              = 0
	MOVE                        = 1
	MOVE_ONTO_STAR              = 2
	MOVE_ONTO_STAR_AND_DIE      = 3
	MOVE_ONTO_STAR_AND_KILL     = 4
	MOVE_ONTO_GLOBE             = 5
	MOVE_ONTO_GLOBE_AND_DIE     = 6
	MOVE_ONTO_ANOTHER_DIE       = 7
	MOVE_ONTO_ANOTHER_KILL      = 8
	MOVE_ONTO_VICTORY_ROAD      = 9
	MOVE_ONTO_GOAL              = 10
	NONE                        = 11

class STATE(IntEnum):

	HOME                        = 0
	GLOBE                       = 1
	STAR                        = 2
	STAR_IN_DANGER              = 3
	COMMON_PATH                 = 4
	COMMON_PATH_WITH_BUDDY      = 5
	COMMON_PATH_CAN_KILL        = 6
	COMMON_PATH_IN_DANGER       = 7
	VICTORY_ROAD                = 8
	GOAL                        = 9

class REWARD(Enum, settings=NoAlias):

	MOVE                        = 5 # proportional to steps taken
	MOVE_FROM_HOME              = 5
	DIE                         = -10
	KILL                        = 5 # + extra proportional to steps taken
	GET_ONTO_VICTORY_ROAD       = 25
	GET_IN_GOAL                 = 50
	WIN                         = 100

class LudoState:

	def __init__(self, state=None, empty=False):

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

	def get_state(self, token_id, player_id=0):

		# return self.get_states()[player_id, token_id]
		x = self.state[player_id, token_id]

		if x == -1:
			return STATE.HOME

		elif x == 99:
			return STATE.GOAL

		elif is_globe_pos(x):
			return STATE.GLOBE

		elif (x > 51) and (x < 99):
			return STATE.VICTORY_ROAD

		elif (x > 0) and (x < 53):
			return STATE.COMMON_PATH

	def get_states(self):

		mat_state = self.state.copy()

		# convert state indicies to STATE descriptor enum indices
		# https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
		for x in np.nditer(mat_state, op_flags=['readwrite']):

			if x == -1:
				x[...] = STATE.HOME.value

			elif x == 99:
				x[...] = STATE.GOAL.value

			elif is_globe_pos(x):
				x[...] = STATE.GLOBE.value

			elif (x > 52) and (x < 99):
				x[...] = STATE.VICTORY_ROAD.value

			elif (x > 0) and (x < 53):
				x[...] = STATE.COMMON_PATH.value

		# create matrix of Enum values
		# https://stackoverflow.com/questions/57907352/have-any-method-to-speed-up-int-list-to-enum
		mat_state_enum = np.array([*STATE],object)[mat_state]

		return mat_state_enum

	def get_reward(self, next_state):
		""" return reward for player 0 (for relative states) """

		LONGEST_STEP = 13

		if will_send_self_home(self, next_state):
			return REWARD.DIE

		elif will_send_opponent_home(self, next_state):
			# return REWARD.KILL
			return REWARD.KILL.value + steps_taken(self, next_state)/LONGEST_STEP * REWARD.MOVE.value

		elif will_send_self_onto_victory_road(self, next_state):
			return REWARD.GET_ONTO_VICTORY_ROAD

		elif will_win_game(next_state):
			return REWARD.WIN

		elif will_send_self_onto_goal(self, next_state):
			return REWARD.GET_IN_GOAL

		elif will_move_from_home(self, next_state):
			return REWARD.MOVE_FROM_HOME

		else:
			return steps_taken(self, next_state)/LONGEST_STEP * REWARD.MOVE.value

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

	def move_token(self, token_id, dice_roll):
		"""
		compute the move for token of player 0 in current state
		return resulting tuple of (new State(), ACTION() taken)
		"""

		cur_pos = self[0][token_id]
		# current_state = self.get_state(token_id)

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
			opponents[opponents == 1] = -1 # kill

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
				return new_state, ACTION.MOVE_ONTO_GLOBE_AND_DIE
			elif (is_globe_pos(target_pos)):
				player[token_id] = target_pos
				return new_state, ACTION.MOVE_ONTO_GLOBE

			# star
			if (star_jump_length := star_jump(target_pos)):

				kill = False
				# check for kills on current star
				if occupant_count == 1:
					opponents[occupants] = -1
					kill = True

				# last star -> send directly to goal
				if target_pos == 51:
					player[token_id] = 99
					return new_state, ACTION.MOVE_ONTO_GOAL

				else:

					target_pos = target_pos + star_jump_length

					occupants = opponents == target_pos
					occupant_count = np.sum(occupants)

					# multiple opponent tokens on ending star
					if occupant_count > 1:
						player[token_id] = -1  # sends self home
						return new_state, ACTION.MOVE_ONTO_STAR_AND_DIE

					# one token on opponent star, kill it
					elif (occupant_count == 1):
						opponents[occupants] = -1
						kill = True

					# perform move
					player[token_id] = target_pos
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
			return new_state, (ACTION.MOVE_ONTO_VICTORY_ROAD if cur_pos < 52 else ACTION.MOVE)

		else:  # bounce back from goal pos
			player[token_id] = 57 - (target_pos - 57)
		return new_state, ACTION.MOVE

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