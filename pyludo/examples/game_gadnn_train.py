from http.client import GATEWAY_TIMEOUT
import sys, os, time, random
import torch, pygad, pygad.torchga
import matplotlib.pyplot as plt
from datetime import datetime

from pyludo.game import LudoGame
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_gadnn import LudoPlayerGADNN
from pyludo.helpers import Tee

if __name__ == "__main__":
	# mp.freeze_support()
	pass

# save files
timestr = datetime.now().strftime("%d%m%Y_%H%M%S")
path = f"data/train_gadnn/{timestr}"
os.makedirs(path)
sys.stdout = Tee(f"{path}/log.txt") # set print() to write to cout/file

# config
SIZE_OBSERVATION_SPACE = 56
SIZE_ACTION_SPACE = 4
NUM_CHROMOSOMES = 20
NUM_GAMES_PER_CHROMOSOME = 50
NUM_GENERATIONS = 200
PARENT_SELECTION_TYPE = ["sss", "rws", "sus", "rank", "random", "tournament"][0]
CROSSOVER_TYPE = ["single_point", "two_points", "uniform", "scattered"][3]
MUTATION_TYPE = ["random", "adaptive"][1]
MUTATION_PROBABILITY = [0.02, [0.05, 0.01]][1] # [good_fit, bad_fit] for "adaptive"
NUM_PARENTS_MATING = NUM_CHROMOSOMES//2 # Number of solutions to be selected as parents in the mating pool.

# create model
torch.set_grad_enabled(False)
model = torch.nn.Sequential(
	# torch.nn.Linear(SIZE_OBSERVATION_SPACE, SIZE_OBSERVATION_SPACE),
	# torch.nn.Sigmoid(),
	torch.nn.Linear(SIZE_OBSERVATION_SPACE, SIZE_ACTION_SPACE),
	torch.nn.Softmax(dim=1)
)

# fitness function
def fitness_func(solution, sol_idx):

	start_time = time.time()

	global model
	model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
	model.load_state_dict(model_weights_dict)

	players = [LudoPlayerGADNN(model), LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom()]
	score_acum = 0
	for _ in range(0, NUM_GAMES_PER_CHROMOSOME):

		random.shuffle(players)
		winner = LudoGame(players).play_full_game()
		score = 1 if players[winner].name == "genetic-algorithm" else 0.01
		score_acum += score

	avg_score = score_acum/NUM_GAMES_PER_CHROMOSOME

	solution_fitness = avg_score
	print(f"Chromosome {sol_idx}: {solution_fitness:.4f} in {time.time() - start_time:.2f} sec")
	return solution_fitness

list_avg_fitnesses = []
def callback_generation(ga_instance):
	global list_avg_fitnesses
	avg_fitness = sum(ga_instance.last_generation_fitness)/ga_instance.sol_per_pop
	list_avg_fitnesses.append(avg_fitness)
	print(f">> Generation {ga_instance.generations_completed}: {avg_fitness:.4f}")

	# save models
	if (ga_instance.generations_completed % 50) == 0:
		print("saving models...")
		ga_instance.save(f"{path}/ga_instance")
		torch.save(model, f"{path}/model.pt")


torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=NUM_CHROMOSOMES)
# initial_population = torch_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       parent_selection_type=PARENT_SELECTION_TYPE,
                       crossover_type=CROSSOVER_TYPE,
                       mutation_type=MUTATION_TYPE,
                       mutation_probability=MUTATION_PROBABILITY,
                       num_parents_mating=NUM_PARENTS_MATING,
                       allow_duplicate_genes=False,
                       initial_population=torch_ga.population_weights,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       save_best_solutions=True,
                       save_solutions=True)

print("starting GA training...")
ga_instance.run()

# save GA instance and pytorch model
ga_instance.save(f"{path}/ga_instance")
torch.save(model, f"{path}/model.pt")

# plot average fitnesses
plt.plot(list_avg_fitnesses)
plt.show()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
# ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)