# utils
from collections import defaultdict

# initialize state for all model parameters
def init_states(params):
    states = dict()
    # create state for each parameter
    for p in params:
        if p not in states:
            states[p] = defaultdict()
    return states