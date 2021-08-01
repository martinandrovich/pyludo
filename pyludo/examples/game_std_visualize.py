import pyglet

from pyludo import LudoGame, LudoVisualizerStep
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive


players = [
	LudoPlayerRandom(),
	LudoPlayerFast(),
	LudoPlayerAggressive(),
	LudoPlayerDefensive(),
]

game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)

print("Use LEFT and RIGHT arrow to progress game.")
pyglet.app.run()
