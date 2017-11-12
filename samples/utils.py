import random
from functools import reduce

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


def generate_word(word, letters, population_size):
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", generate_letter, letters=letters)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(word))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_alphabet_distance, word=word, alphabet=letters)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutInAlphabet, letters=letters, max_step=3, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    generation = 0
    while have_not_found_yet(population, generation, word):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        generation += 1
        if generation % 100 == 0:
            print(generation)

    print(generation)
    print(tools.selBest(population, k=1)[0])


def have_not_found_yet(population, generation, word):
    found = False
    if (generation != 0):
        for idx, letter in enumerate(tools.selBest(population, k=1)[0]):
            if (letter != word[idx]):
                found = True
    else:
        found = True
    return found


def mutInAlphabet(individual, letters, max_step, indpb):
    letter_index_in_word = random.randint(0, len(individual) - 1)
    letter_index_in_alphabet = letters.index(individual[letter_index_in_word])
    step = random.randint(0, max_step)

    if random.random() < indpb:
        if random.random() > 0.5:
            if letter_index_in_alphabet + step < len(letters):
                individual[letter_index_in_word] = letters[letter_index_in_alphabet + step]
        else:
            if letter_index_in_alphabet - step >= 0:
                individual[letter_index_in_word] = letters[letter_index_in_alphabet - step]

    return individual,


def evaluate_alphabet_distance(individual, word, alphabet):
    score = 0
    for idx, letter in enumerate(individual):
        score += abs(alphabet.index(word[idx]) - alphabet.index(letter))
    return score,


def generate_letter(letters):
    return letters[random.randint(0, 25)]


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
