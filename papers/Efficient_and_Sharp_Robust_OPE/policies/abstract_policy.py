from abc import ABC, abstractmethod
import gymnasium as gym


class AbstractPolicy(ABC):
    def __init__(self, env):
        assert isinstance(env, gym.Env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def __call__(self, s):
        assert self.observation_space.contains(s)
        a = self.sample_action(s)
        # print(a, self.action_space)
        assert self.action_space.contains(a)
        return a

    @abstractmethod
    def sample_action(self, s):
        pass
