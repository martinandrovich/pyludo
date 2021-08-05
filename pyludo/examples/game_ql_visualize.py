import pyglet
import logging
import numpy as np

from pyludo import LudoGame, LudoVisualizerStep
from pyludo.players import LudoPlayerRandom
from pyludo.player_ql import LudoPlayerQLearning

logging.basicConfig(level=logging.INFO, force=True)

qtable = np.loadtxt("data/qtable.csv", delimiter=",")

# create players
p = LudoPlayerQLearning(training=True,
                        decaying_epsilon=False,
                        qtable=qtable)

players = [
	p,
	p,
	LudoPlayerRandom(),
	LudoPlayerRandom(),
]

game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)

print("Use LEFT and RIGHT arrow to progress game.")
pyglet.app.run()
