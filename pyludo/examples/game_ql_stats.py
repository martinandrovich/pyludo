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

SESSION_ID = "train_sim_std"
DATA_DIR = f"data/test_{SESSION_ID}"
os.mkdir(DATA_DIR)

# create players
players = [
	LudoPlayerQLearning(qtable="data/train_sim/qtable.csv", advanced=False),
	# LudoPlayerRandom(),
	# LudoPlayerRandom(),
	# LudoPlayerRandom(),
	# LudoPlayerRandom(),
	LudoPlayerFast(),
	LudoPlayerAggressive(),
	LudoPlayerDefensive(),
]

# assign ID's to players
for i, p in enumerate(players):
	p.id = i

# write player info to file
# players[isinstance(players, LudoPlayerQLearning)].save_info(DATA_DIR)

# statistics array
stats = []

# start games
start_time = time.time()
for test in range(NUM_TESTS):

	wl_avg = 0.
	scores = {}
	for player in players:
		scores[player.name] = 0

	for episode in range(NUM_MAX_EPISODES):

		# shuffle players and start game
		random.shuffle(players)
		ludoGame = LudoGame(players)
		winner = ludoGame.play_full_game()

		scores[players[winner].name] += 1
		# wl = (1 if players[winner].name == "q-learning" else 0)
		wl = (1 if players[winner].id == 0 else 0)
		wl_avg = (wl + episode * wl_avg)/(episode + 1)
		print(f"Game {episode}/{NUM_MAX_EPISODES} @ {(NUM_MAX_EPISODES * test + episode)/(time.time() - start_time)} game/s")

	duration = time.time() - start_time
	stats.append({ "W/L": wl_avg,
	               "Distribution": scores })


# results
print("\nRESULTS:\n")
for dict in stats:
	print(dict)

# final W/L
wl_final = sum(item["W/L"] for item in stats)/len(stats)
print(f"Final average W/L: {wl_final}")

# test info
info = { "players": [( p.name + (f" ({p.qtable_path})" if hasattr(p, "qtable_path") else "") ) for p in players],
         "num_episodes": NUM_MAX_EPISODES,
         "num_tests": NUM_TESTS,
         "wl_final": wl_final,
}

# write info to file
with open(f"{DATA_DIR}/test_info.txt", "a") as fs:
	for key, value in info.items():
		fs.write(f"{key}: {value}\n")

# write stats to file(s)
with open(f"{DATA_DIR}/wl.csv", "a") as fs_wl, open(f"{DATA_DIR}/dist.csv", "a") as fs_dist:
	for i, item in enumerate(stats):
		fs_wl.write(f"{i}, {item['W/L']}\n")
		fs_dist.write(f"{i}, {item['Distribution']}\n")