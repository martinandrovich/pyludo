import time
import random
import numpy as np
import os
import copy

from pyludo.game import LudoGame
from pyludo.state import STATE, ACTION
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_ql import LudoPlayerQLearning
from pyludo.helpers import running_avg

# config
NUM_MAX_EPISODES = 20_000
qtable_diff = np.inf
qtable_prev = None

# data output
SESSION_ID = "train_adv_cont"
DATA_DIR = f"data/{SESSION_ID}"
os.mkdir(DATA_DIR)

fs_qtable_diff = open(f"{DATA_DIR}/qtable_diff.csv", "a")
fs_wl_avg = open(f"{DATA_DIR}/wl_avg.csv", "a")
fs_epsilon = open(f"{DATA_DIR}/epsilon.csv", "a")
fs_cum_reward = open(f"{DATA_DIR}/cum_reward.csv", "a")

# create new Q-table
qtable = np.loadtxt(f"data/train_adv/qtable.csv", delimiter=",")
# qtable = np.zeros((len(STATE), len(ACTION)), dtype=float)
np.savetxt(f"{DATA_DIR}/qtable.csv", qtable, delimiter=",", fmt="%f")
qtable = np.loadtxt(f"{DATA_DIR}/qtable.csv", delimiter=",")

# Q-learning player(s)

p1 = LudoPlayerQLearning(
    qtable=qtable,
    training=True,
    advanced=True, # simple or advanced SS
    num_max_episodes=NUM_MAX_EPISODES,
    decaying_epsilon=True,
    epsilon_max=0.30,
    epsilon_min=0.01,
    alpha=0.001,
    gamma=0.9,
)

p2 = copy.copy(p1)
                         
# save initial training info
p1.save_info(DATA_DIR)

# create players
players = [
	p1,
	p2,
	# LudoPlayerRandom(),
	# LudoPlayerRandom(),
	LudoPlayerAggressive(),
	LudoPlayerDefensive(),
]

scores = {}
for player in players:
	scores[player.name] = 0

# start games
start_time = time.time()
for i in range(NUM_MAX_EPISODES):

	# prime Q-learning player for training
	p1.new_episode(), p2.new_episode()
	qtable_prev = p1.qtable.copy()
	
	# shuffle players and start game
	random.shuffle(players)
	ludoGame = LudoGame(players)
	winner = ludoGame.play_full_game()
	
	# stats
	scores[players[winner].name] += 1
	wl_avg = running_avg(1 if players[winner] is p1 else 0, 100)
	# wl_avg = running_avg(1 if (players[winner] is p1 or players[winner] is p2) else 0, 100)
	qtable_diff = np.abs(np.sum(qtable_prev) - np.sum(p1.qtable))

	# log data	
	out = [f"Game {i}/{NUM_MAX_EPISODES} @ {i/(time.time() - start_time)} game/s",
	       f"W/L: {wl_avg}",
	       f"delta: {qtable_diff}",
	       f"epsilon: {p1.epsilon}",
	       f"reward: {p1.cum_reward}\n",
	]
	print("\n".join(out))
	
	if (i % 10) == 0:
		p1.save_qtable(f"{DATA_DIR}/qtable.csv")
		fs_qtable_diff.write(f"{i},{qtable_diff}\n")
		fs_wl_avg.write(f"{i},{wl_avg}\n")
		fs_epsilon.write(f"{i},{p1.epsilon}\n")
		fs_cum_reward.write(f"{i},{p1.cum_reward}\n")

duration = time.time() - start_time

fs_qtable_diff.close()
fs_wl_avg.close()
fs_epsilon.close()

print('win distribution:', scores)
print('games per second:', NUM_MAX_EPISODES / duration)
