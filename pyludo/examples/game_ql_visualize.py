import pyglet
import logging
import numpy as np

from pyludo import LudoGame, LudoVisualizerStep
from pyludo.players import LudoPlayerRandom, LudoPlayerHuman
from pyludo.player_ql import LudoPlayerQLearning

logging.basicConfig(level=logging.INFO, force=True)

players = [
	LudoPlayerQLearning(qtable="data/train_2/qtable.csv", training=True),
	LudoPlayerRandom(),
	LudoPlayerRandom(),
	LudoPlayerRandom(),
]

game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)

print("Use LEFT and RIGHT arrow to progress game.")
pyglet.app.run()
