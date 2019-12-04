import numpy as np


def get_num_grid_choices(grid):
    bases = []
    for k, v in list(grid.items()):
        for v_in in list(v.values()):
            bases.append(len(v_in))
    bases = np.array(bases)
    return bases, np.prod(bases)

def grid_choices(grid, bases):
    # Grab choices generator
    increment = _next_choice(bases)
    # Unflatten
    for choices in increment:
        index = 0
        constants = {}
        for k, v in list(grid.items()):
            constants[k] = {}
            for k_in, v_in in list(v.items()):
                constants[k][k_in] = v_in[choices[index]]
                index += 1
        yield constants

def _next_choice(bases):
    ret = np.zeros(shape=(bases.shape[0],), dtype=np.int)
    yield ret
    while not np.array_equal(ret, bases-1):
        i = 0
        overflow = True
        while overflow:
            overflow = True if ((ret[i] + 1) % bases[i]) == 0 else False
            if not overflow: ret[i] += 1
            else: ret[i] = 0; i += 1
        yield ret
