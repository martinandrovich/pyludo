import sys, os, time, random, json
import pygad
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from pyludo.game import LudoGame
from pyludo.players import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
from pyludo.player_ga import LudoPlayerGA
from pyludo.helpers import Tee
from pyludo.state import ACTION

if __name__ == "__main__":
	# mp.freeze_support()
	pass

# save files
timestr = datetime.now().strftime("%d%m%Y_%H%M%S")
path = f"data/train_ga/{timestr}"
os.makedirs(path)
sys.stdout = Tee(f"{path}/log.txt") # set print() to write to cout/file

# config
NUM_CHROMOSOMES = 20
NUM_GENES = len(ACTION) # type: ignore
GENE_TYPE = float
GENE_SPACE = {"low": 0, "high": 100}
NUM_GAMES_PER_CHROMOSOME = 50
NUM_GENERATIONS = 100
PARENT_SELECTION_TYPE = ["sss", "rws", "sus", "rank", "random", "tournament"][0]
CROSSOVER_TYPE = ["single_point", "two_points", "uniform", "scattered"][0]
MUTATION_TYPE = ["random", "adaptive"][0]
MUTATION_PROBABILITY = [0.05, [0.05, 0.01]][0] # [good_fit, bad_fit] for "adaptive"
NUM_PARENTS_MATING = NUM_CHROMOSOMES//2 # Number of solutions to be selected as parents in the mating pool.

# fitness function
def fitness_func(solution, sol_idx):

	start_time = time.time()

	players = [LudoPlayerGA(weights=solution), LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom()]
	score_acum = 0
	for _ in range(0, NUM_GAMES_PER_CHROMOSOME):

		# random.shuffle(players)
		winner = LudoGame(players).play_full_game()
		score = 1 if players[winner].name == "genetic-algorithm" else 0.01
		score_acum += score

	avg_score = score_acum/NUM_GAMES_PER_CHROMOSOME

	solution_fitness = avg_score
	print(f"Chromosome {sol_idx}: {solution_fitness:.4f} in {time.time() - start_time:.2f} sec")
	return solution_fitness

lst_fit_avg, lst_fit_top, lst_fit_best = [], [], []
def callback_generation(ga_instance):

	fit_avg = np.mean(ga_instance.last_generation_fitness)
	fit_top = np.mean(-np.partition(-ga_instance.last_generation_fitness, 5)[:5]) # top 5
	fit_best = np.max(ga_instance.last_generation_fitness)

	global lst_fit_avg, lst_fit_top, lst_fit_best
	lst_fit_avg.append(fit_avg)
	lst_fit_top.append(fit_top)
	lst_fit_best.append(fit_best)

	print(f"\n>> Generation {ga_instance.generations_completed}"
	      f"\n>> Fitnesses: {ga_instance.last_generation_fitness}"
	      f"\n>> Average: {fit_avg:.4f}, Top 5: {fit_top:.4f}, Best: {fit_best:.4f}\n"
	)

	# save models
	if (ga_instance.generations_completed % 50) == 0:
		print("saving models...")
		ga_instance.save(f"{path}/ga_instance")

def callback_fitness(ga_instance, population_fitness):
	# avg_fitness = np.mean(population_fitness)
	# print(population_fitness)
	pass

ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       sol_per_pop=NUM_CHROMOSOMES,
                       num_genes=NUM_GENES,
                       gene_type=GENE_TYPE,
                       gene_space=GENE_SPACE,
                       parent_selection_type=PARENT_SELECTION_TYPE,
                       crossover_type=CROSSOVER_TYPE,
                       mutation_type=MUTATION_TYPE,
                       mutation_probability=MUTATION_PROBABILITY,
                       num_parents_mating=NUM_PARENTS_MATING,
                       keep_parents=0, # enforce re-calculation of fitness each gen
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                    #    on_fitness=callback_fitness,
                       save_best_solutions=True,
                       save_solutions=True)

# dump info
json.dump(vars(ga_instance), fp=open(f"{path}/info.json", "w"), default=lambda obj: str(type(obj)), indent="\t", ensure_ascii=False)

# train
print("starting GA training...")
ga_instance.run()

# save GA instance + data
ga_instance.save(f"{path}/ga_instance")
json.dump(lst_fit_avg, fp=open(f"{path}/lst_fit_avg.json", "w"), indent="\t", ensure_ascii=False)
json.dump(lst_fit_top, fp=open(f"{path}/lst_fit_top.json", "w"), indent="\t", ensure_ascii=False)
json.dump(lst_fit_best, fp=open(f"{path}/lst_fit_best.json", "w"), indent="\t", ensure_ascii=False)
json.dump(vars(ga_instance), fp=open(f"{path}/info.json", "w"), default=lambda obj: str(type(obj)), indent="\t", ensure_ascii=False)

# plot fitnesses
fig, axs = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
axs[0].plot(lst_fit_avg); axs[0].set_title("Average fitness")
axs[1].plot(lst_fit_top); axs[1].set_title("Top fitness")
axs[2].plot(lst_fit_best); axs[2].set_title("Best fitness")
fig.savefig(f"{path}/plot.png")
plt.show()