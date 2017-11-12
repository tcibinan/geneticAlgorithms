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

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

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


def min(func, search_region, population_size, generations, invert=False):
    creator.create("Fitness", base.Fitness, weights=(1.0 if invert else -1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", float_between, *search_region)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda individual: (func(individual[0]),))
    toolbox.register("mate", skippingCrosover)
    (a, b) = search_region
    toolbox.register("mutate", mutInRegion, a=a, b=b, max_step=0.1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    print(tools.selBest(population, k=1))

    base_range = float_range(*search_region)
    plot(base_range, [func(x) for x in base_range], 'r')
    show()


def inflection_point(func, search_region, population_size, generations):
    geneticAlgorithmWithEvalFunctional(func, evaluate_diff, search_region, population_size, generations,
                                       func=func,
                                       step=0.001)


def zero(func, search_region, population_size, generations):
    geneticAlgorithmWithEvalFunctional(func, evaluate_zero, search_region, population_size, generations,
                                       func=func)


def evaluate_zero(individual, func):
    return abs(func(individual[0])),


def geneticAlgorithmWithEvalFunctional(base_func, eval_func, search_region, population_size, generations, **args):
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", float_between, *search_region)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func, **args)
    toolbox.register("mate", skippingCrosover)
    (a, b) = search_region
    toolbox.register("mutate", mutInRegion, a=a, b=b, max_step=0.1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    print(tools.selBest(population, k=1))

    base_range = float_range(*search_region)
    plot(base_range, [base_func(x) for x in base_range], 'r')
    show()


def evaluate_diff(individual, func, step):
    y = func(individual[0])
    left = abs(y - func(individual[0] - step))
    right = abs(y - func(individual[0] + step))
    return (right if right < left else left),


def max(func, search_region, population_size, generations):
    min(func, search_region, population_size, generations, invert=True)


def show_best_individual(base_range, func, population):
    best_individual = tools.selBest(population, k=1)[0]
    plot(base_range, best_individual, 'g')
    plot(base_range, [func(x) for x in base_range], 'r')
    show()


def float_between(lower_bound, upper_bound):
    return random.random() * (upper_bound - lower_bound) + lower_bound


def float_range(a, b, step=0.5):
    return np.arange(a, b, step)


def skippingCrosover(ind1, ind2):
    return (ind1, ind2,)


def getMu(a, b):
    return (a + b) / 2


def mutInRegion(individual, a, b, max_step, indpb):
    step = random.random() * max_step

    if random.random() < indpb:
        if random.random() > 0.5:
            if individual[0] + step < b:
                individual[0] += step
        else:
            if individual[0] - step > a:
                individual[0] -= step

    return individual,
