import random
import time

from pyludo import LudoGame, LudoPlayerRandom

# config
NUM_MAX_EPISODES = 1000

players = [LudoPlayerRandom() for _ in range(4)]
for i, player in enumerate(players):
	player.id = i

score = [0, 0, 0, 0]

start_time = time.time()
for episode in range(NUM_MAX_EPISODES):

	random.shuffle(players)
	ludoGame = LudoGame(players)
	winner = ludoGame.play_full_game()
	score[players[winner].id] += 1
	print(f"Game {episode}/{NUM_MAX_EPISODES} @ {(NUM_MAX_EPISODES * 0 + episode)/(time.time() - start_time)} game/s")
	
duration = time.time() - start_time

print('win distribution:', score)
