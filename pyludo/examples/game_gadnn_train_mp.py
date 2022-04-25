from glob import glob
import time
import random
import numpy as np
import torch
import pygad
import pygad.torchga

from concurrent import futures
import multiprocessing as mp
import matplotlib.pyplot as plt

from pyludo.game import LudoGame
from pyludo.player_gadnn import LudoPlayerGADNN
from pyludo.state import STATE, ACTION
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_gadnn import LudoPlayerGADNN

# multiprocessing version - only on LINUX
# https://hackernoon.com/how-genetic-algorithms-can-compete-with-gradient-descent-and-backprop-9m9t33bq

if __name__ == "__main__":
	# mp.freeze_support() # for debugging
	pass

# config
NUM_CHROMOSOMES = 10
NUM_PARENTS_MATING = 5 # Number of solutions to be selected as parents in the mating pool.
NUM_GENERATIONS = 200
NUM_GAMES_PER_CHROMOSOME = 100
NUM_THREADS = 4
NUM_INPUTS = 20
NUM_ACTIONS = 4

# create model
torch.set_grad_enabled(False)
model = torch.nn.Sequential(
	torch.nn.Linear(NUM_INPUTS, 100),
	torch.nn.ReLU(),
	torch.nn.Linear(100, 200),
	torch.nn.ReLU(),
	torch.nn.Linear(200, NUM_ACTIONS),
	torch.nn.Softmax(dim=1)
)
model.eval()

def play_game(solution):
	players = [LudoPlayerGADNN(model, solution), LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom()]
	random.shuffle(players)
	winner = LudoGame(players).play_full_game()
	score = 1 if isinstance(players[winner], LudoPlayerGADNN) else 0.01
	return score

# fitness function
def fitness_func(solution, sol_idx):
	global model
	start_time = time.time()

	# multi-threaded
	# scores = []
	# with mp.Pool(processes=6) as pool:
	# 	scores = pool.map(play_game, [solution for _ in range(NUM_GAMES_PER_CHROMOSOME)])
	# avg_score = sum(scores)/NUM_GAMES_PER_CHROMOSOME

	# single thread
	players = [LudoPlayerGADNN(None, None), LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom()]
	running_score = 0
	for i in range(0, NUM_GAMES_PER_CHROMOSOME):

		# overwrite player
		for i in range(0, len(players)):
			if isinstance(players[i], LudoPlayerGADNN): players[i] = LudoPlayerGADNN(model, solution)

		random.shuffle(players)
		winner = LudoGame(players).play_full_game()
		score = 1 if isinstance(players[winner], LudoPlayerGADNN) else 0.01
		running_score += score

	avg_score = running_score/NUM_GAMES_PER_CHROMOSOME

	solution_fitness = avg_score
	# solution_fitness = 1.0 / avg_score
	print(f"Chromosome {sol_idx}: {solution_fitness:.4f} in {time.time() - start_time:.2f} sec")
	return solution_fitness


def fitness_wrapper(solution):
	return fitness_func(solution, 0)

list_avg_fitnesses = []
def callback_generation(ga_instance):
	global list_avg_fitnesses
	avg_fitness = sum(ga_instance.last_generation_fitness)/ga_instance.sol_per_pop
	list_avg_fitnesses.append(avg_fitness)
	print(f">> Generation {ga_instance.generations_completed}: {avg_fitness:.4f}")

class PooledGA(pygad.GA):

	def cal_pop_fitness(self):
		global pool
		pop_fitness = pool.map(fitness_wrapper, self.population)
		print(pop_fitness)
		pop_fitness = np.array(pop_fitness)
		return pop_fitness

with mp.Pool(processes=4) as pool:

	# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
	torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=NUM_CHROMOSOMES)
	# initial_population = torch_ga.population_weights # Initial population of network weights

	# ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
	#                        num_parents_mating=NUM_PARENTS_MATING,
	#                        initial_population=torch_ga.population_weights,
	#                        fitness_func=fitness_func,
	#                        on_generation=callback_generation)

	ga_instance = PooledGA(num_generations=NUM_GENERATIONS,
	                       num_parents_mating=NUM_PARENTS_MATING,
	                       initial_population=torch_ga.population_weights,
	                       fitness_func=fitness_func,
	                       on_generation=callback_generation)

	# start training
	print("starting GA training...")
	ga_instance.run()

	# plot average fitnesses
	plt.plot(list_avg_fitnesses)
	plt.show()

	# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
	ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

	# Returning the details of the best solution.
	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
	print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))