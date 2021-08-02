import pyglet

from pyludo import LudoGame, LudoVisualizerStep
from pyludo.players import LudoPlayerRandom
from pyludo.player_ql import LudoPlayerQLearning


players = [
	LudoPlayerQLearning(training=True),
	LudoPlayerRandom(),
	LudoPlayerRandom(),
	LudoPlayerRandom(),
]

game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)

print("Use LEFT and RIGHT arrow to progress game.")
pyglet.app.run()
