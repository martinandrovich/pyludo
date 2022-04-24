import torch, pygad, pygad.torchga

from pyludo.game import LudoGame
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_ga import LudoPlayerGA

# path to model
timestr = "23042022_132706"
path = f"data/train_ga/{timestr}"

# load model and set to eval() mode
model = torch.load(f"{path}/model.pt")
model.eval()

# load GA
ga_instance = pygad.load(f"{path}/ga_instance.ga")

# find best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# evaluate model with GA player vs 3 random players
model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
model.load_state_dict(model_weights_dict)

num_games = 100
players = [LudoPlayerGA(model), LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom()]
wl_acum = 0
for i in range(num_games):
	# random.shuffle(players)
	winner = LudoGame(players).play_full_game()
	score = 1 if players[winner].name == "genetic-algorithm" else 0
	wl_acum += score
wl_avg = wl_acum/num_games

print(f"average win-rate: {wl_avg:.4f}")