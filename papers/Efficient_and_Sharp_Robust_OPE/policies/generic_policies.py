import numpy as np
from policies.abstract_policy import AbstractPolicy


class UniformPolicy(AbstractPolicy):
    def __init__(self, env):
        super().__init__(env)

    def sample_action(self, s):
        return self.action_space.sample()


class EpsilonSmoothPolicy(AbstractPolicy):
    def __init__(self, env, pi_base, epsilon):
        super().__init__(env)
        self.pi_base = pi_base
        self.epsilon = epsilon

    def sample_action(self, s):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.pi_base(s)
        