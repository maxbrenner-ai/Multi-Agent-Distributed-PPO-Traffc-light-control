import random


def grid_choices_random(grid, num_runs):
    # Unflatten
    possibilities = []
    for v in list(grid.values()):
        for v_in in list(v.values()):
            possibilities.append(v_in)
    incrementer = _next_choice(possibilities, num_runs)

    for choices in incrementer:
        constants = {}
        index = 0
        for k, v in list(grid.items()):
            constants[k] = {}
            for k_in in list(v.keys()):
                constants[k][k_in] = choices[index]
                index += 1
        yield constants

def _get_random_choice(possibilities):
    choice = []
    for const in possibilities:
        choice.append(random.sample(const, 1)[0])
    return choice

def _next_choice(possibilities, num_runs):
    storage = []
    index = 0
    while index < num_runs:
        choice = _get_random_choice(possibilities)
        while choice in storage:
            choice = _get_random_choice(possibilities)
        storage.append(choice)
        index += 1
        yield choice
