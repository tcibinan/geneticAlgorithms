from .utils import approximate, float_range

print('y = x')
approximate(
    func=lambda x: x,
    base_range=float_range(-2.0, 2.0, 0.5),
    bounds=(-10.0, 10.0,),
    population_size=300,
    generations=50)