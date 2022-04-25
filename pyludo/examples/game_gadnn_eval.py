import torch, pygad, pygad.torchga

from pyludo.game import LudoGame
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_gadnn import LudoPlayerGADNN

# path to model
timestr = "24042022_142721"
path = f"data/train_gadnn/{timestr}"

# load model and set to eval() mode
model = torch.load(f"{path}/model.pt")
model.eval()

# load GA (cannot evaluate fitness function)
fitness_func = callback_generation = None
ga_instance = pygad.load(f"{path}/ga_instance")

# find best solution
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

best_gen = ga_instance.best_solution_generation
best_sol = ga_instance.best_solutions[200]

# evaluate model with GA player vs 3 random players
model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=best_sol)
model.load_state_dict(model_weights_dict)

num_games = 1000
players = [LudoPlayerGADNN(model), LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom()]
wl_acum = 0
for i in range(num_games):
	# random.shuffle(players)
	winner = LudoGame(players).play_full_game()
	score = 1 if players[winner].name == "genetic-algorithm" else 0
	wl_acum += score
wl_avg = wl_acum/num_games

print(f"average win-rate: {wl_avg:.4f}")