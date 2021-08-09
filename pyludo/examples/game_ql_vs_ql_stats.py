import time
import random
import os
import numpy as np

from pyludo.game import LudoGame
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_ql import LudoPlayerQLearning

# config
NUM_MAX_EPISODES = 1000
NUM_TESTS = 100

SESSION_NUMBER = 10
DATA_DIR = f"data/test_{SESSION_NUMBER}"
os.mkdir(DATA_DIR)

QLS = "q-learning-simple"
QLA = "q-learning-advanced"

# create players
players = [
	LudoPlayerQLearning(qtable="data/train_0/qtable.csv", name=QLS, advanced=False),
	LudoPlayerQLearning(qtable="data/train_0/qtable.csv", name=QLS, advanced=False),
	LudoPlayerQLearning(qtable="data/train_6/qtable.csv", name=QLA, advanced=True),
	LudoPlayerQLearning(qtable="data/train_6/qtable.csv", name=QLA, advanced=True),
	# LudoPlayerRandom(),
	# LudoPlayerRandom(),
]

# statistics array
stats = []

# start games
start_time = time.time()
for test in range(NUM_TESTS):

	wl_sim_avg = 0.
	wl_adv_avg = 0.
	scores = {}
	for player in players:
		scores[player.name] = 0

	for episode in range(NUM_MAX_EPISODES):

		# shuffle players and start game
		random.shuffle(players)
		ludoGame = LudoGame(players)
		winner = ludoGame.play_full_game()

		scores[players[winner].name] += 1
		wl_sim = (1 if players[winner].name == QLS else 0)
		wl_adv = (1 if players[winner].name == QLA else 0)
		wl_sim_avg = (wl_sim + episode * wl_sim_avg)/(episode + 1)
		wl_adv_avg = (wl_adv + episode * wl_adv_avg)/(episode + 1)
		
		print(f"Game {episode}/{NUM_MAX_EPISODES} @ {(NUM_MAX_EPISODES * test + episode)/(time.time() - start_time)} game/s")

	duration = time.time() - start_time
	stats.append({ "W/L": [wl_sim_avg, wl_adv_avg],
	               "Distribution": scores })


# results
print("\nRESULTS:\n")
for dict in stats:
	print(dict)

# final W/L
wl_sim_final = sum(item["W/L"][0] for item in stats)/len(stats)
wl_adv_final = sum(item["W/L"][1] for item in stats)/len(stats)
print(f"Final average W/L: [{wl_sim_final}, {wl_adv_final}]")

# test info
info = { "players": [( p.name + (f" ({p.qtable_path})" if hasattr(p, "qtable_path") else "") ) for p in players],
         "num_episodes": NUM_MAX_EPISODES,
         "num_tests": NUM_TESTS,
         "wl_final": [{wl_sim_final}, {wl_adv_final}],
}

# write info to file
with open(f"{DATA_DIR}/test_info.txt", "a") as fs:
	for key, value in info.items():
		fs.write(f"{key}: {value}\n")

# write stats to file(s)
with open(f"{DATA_DIR}/wl.csv", "a") as fs_wl, open(f"{DATA_DIR}/dist.csv", "a") as fs_dist:
	for i, item in enumerate(stats):
		fs_wl.write(f"{i}, {item['W/L'][0]}, {item['W/L'][1]}\n")
		fs_dist.write(f"{i}, {item['Distribution']}\n")