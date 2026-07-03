import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    values = np.asarray(values)
    transitions = np.asarray(transitions)
    rewards = np.asarray(rewards)
    result = np.sum(transitions * values, axis = 2)
    result = result * gamma 
    result = rewards + result
    max_values = np.max(result, axis = 1)
    return max_values.tolist()