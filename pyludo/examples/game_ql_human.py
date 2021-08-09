import pyglet
import logging
import numpy as np

from pyludo import LudoGame, LudoVisualizerStep
from pyludo.players import LudoPlayerRandom, LudoPlayerHuman
from pyludo.player_ql import LudoPlayerQLearning

logging.basicConfig(level=logging.INFO, force=True)

qtable = np.loadtxt("data/train_1/qtable.csv", delimiter=",")

players = [
	LudoPlayerQLearning(qtable=qtable, advanced=True),
	LudoPlayerQLearning(qtable=qtable, advanced=True),
	LudoPlayerQLearning(qtable=qtable, advanced=True),
	LudoPlayerHuman(advanced=False),
]

game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)

print("Use LEFT and RIGHT arrow to progress game.")
pyglet.app.run()
