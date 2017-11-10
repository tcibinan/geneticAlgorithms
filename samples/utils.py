import random
import numpy as np

from deap import base, creator, tools, algorithms
from matplotlib.pyplot import plot, show


def approximate(func, base_range, bounds, population_size, generations):
    def evaluation_func(individual):
        score = 0.0
        for idx, x in enumerate(base_range):
            score += abs(func(x) - individual[idx])
        return score,

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", float_between, *bounds)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(base_range))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluation_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    show_best_individual(base_range, func, population)


def show_best_individual(base_range, func, population):
    best_individual = tools.selBest(population, k=1)[0]
    plot(base_range, best_individual, 'g')
    plot(base_range, [func(x) for x in base_range], 'r')
    show()


def float_between(lower_bound, upper_bound):
    return random.random() * (upper_bound - lower_bound) + lower_bound


def float_range(a, b, step=0.5):
    return np.arange(a, b, step)
