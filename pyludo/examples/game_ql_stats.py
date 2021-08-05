import time
import random
import numpy as np

from pyludo import LudoGame
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_ql import LudoPlayerQLearning
from pyludo.helpers import running_avg

# config
num_max_episodes = 100

# Q-learning player
p = LudoPlayerQLearning(training=False, qtable="data/qtable.csv")

# create players
players = [
	p,
	p,
	LudoPlayerRandom(),
	LudoPlayerRandom(),
]

scores = {}
for player in players:
	scores[player.name] = 0

# start games
start_time = time.time()
for i in range(num_max_episodes):
	
	# shuffle players and start game
	random.shuffle(players)
	ludoGame = LudoGame(players)
	winner = ludoGame.play_full_game()
	
	scores[players[winner].name] += 1
	wl_avg = running_avg(1 if players[winner] is p else 0, 100)
	print(f"Game {i}/{num_max_episodes}")
	print(f"W/L: {wl_avg}")

duration = time.time() - start_time

print('win distribution:', scores)
print('games per second:', num_max_episodes / duration)
