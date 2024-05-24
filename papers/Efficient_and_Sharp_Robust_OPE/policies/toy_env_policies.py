from policies.abstract_policy import AbstractPolicy
from environments.toy_env import ToyEnv

"""
Contains policies specific to ToyEnv
"""

class ThresholdPolicy(AbstractPolicy):
    def __init__(self, env, s_threshold):
        super().__init__(env)
        assert isinstance(env, ToyEnv)
        self.s_threshold = s_threshold
        self.pass_action = env.PASS_ACTION
        self.control_action = env.CONTROL_ACTION

    def sample_action(self, s):
        if s[0] < self.s_threshold:
            return self.pass_action
        else:
            return self.control_action
