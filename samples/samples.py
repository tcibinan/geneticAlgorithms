from math import exp, log

from utils import approximate, float_range, min, max, inflection_point, zero

a = 4
b = 5

print('y = %d*x+%d' % (a, b,))
approximate(
    func=lambda x: a * x + b,
    base_range=float_range(-2.0, 2.0, 0.5),
    bounds=(-10.0, 10.0,),
    population_size=300,
    generations=50)

print('y = %d*exp(x*%f)' % (a, b,))
approximate(
    func=lambda x: a * exp(x * b),
    base_range=float_range(-2.0, 2.0, 0.5),
    bounds=(-10000.0, 10000.0,),
    population_size=500,
    generations=50)

min(
    func=lambda x: (x-0.8)**2 + 4,
    search_region=(-2.0, 2.0,),
    population_size=300,
    generations=50
)

max(
    func=lambda x: 1.0/x,
    search_region=(-4.0, -0.0,),
    population_size=300,
    generations=50
)

inflection_point(
    func=lambda x: (x - 1.5) ** 3 + 3,
    search_region=(-10.0, 10.0,),
    population_size=300,
    generations=50
)

zero(
    func=lambda x: log(x + 1) - 2.25,
    search_region=(-0.5, 15.0,),
    population_size=300,
    generations=50
)

