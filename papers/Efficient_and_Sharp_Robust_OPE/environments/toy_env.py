import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import torch
import torch.nn.functional as F
import numpy as np

register(
    id = "ToyEnv",
    entry_point = "environments.toy_env:ToyEnv"
)

class ToyEnv(gym.Env):
    def __init__(self, s_init=1.0, adversarial=True, adversarial_lambda=2.0):
        super(ToyEnv, self).__init__()
        self.s = None

        self.s_init = s_init
        self.s_max_val = 5.0
        self.adversarial = adversarial
        self.adversarial_lambda = adversarial_lambda

        self.control_frac= 0.2

        self.drift_min = -0.2
        self.drift_max = 1.0

        self.num_init_basis_func = 20

        self.control_drift_min = -0.1
        self.control_drift_max = 0.5

        # self.max_risk = self.s_max_val ** 2
        # self.max_risk = self.s_max_val
        self.control_cost = 1.0
        self.max_risk = (self.s_max_val ** 2) + self.control_cost

        self.observation_space = spaces.Box(low=0, high=self.s_max_val)
        self.action_space = spaces.Discrete(2)

        self.PASS_ACTION = 0
        self.CONTROL_ACTION = 1
        self.ACTION_IDX_TO_NAME = {
            self.PASS_ACTION: "pass",
            self.CONTROL_ACTION: "control",
        }

    def reset(self, seed=None, options=None):
        self.s = self.s_init
        return np.array([self.s], dtype="float32"), {}

    def step(self, action):
        # first decide if we are doing regular or adversarial transition

        if not self.adversarial:
            s_updated = self._transition_regular(action)
        else:
            prob_adversarial = 1 - 1 / self.adversarial_lambda
            if np.random.rand() > prob_adversarial:
                s_updated = self._transition_regular(action)
            else:
                s_updated = self._transition_adversarial(action)

        # risk = (self.s ** 2 + s_updated ** 2 + self.s * s_updated) / 3.0
        # risk = (self.s + s_updated) / 2.0
        is_control = (action == self.CONTROL_ACTION)
        risk = self.s ** 2 + self.control_cost * is_control
        reward = (self.max_risk - risk) / self.max_risk
        self.s = s_updated
        return np.array([s_updated], dtype="float32"), reward, False, False, {}

    def _get_transition_min_max(self, action):
        assert action in self.ACTION_IDX_TO_NAME
        a_name = self.ACTION_IDX_TO_NAME[action]

        # get normal range for non-adversarial transition
        if a_name == "control":
            s_min = self.control_frac * (self.s + self.control_drift_min)
            s_max = self.s + self.control_drift_max
        elif a_name == "pass":
            s_min = self.s + self.drift_min
            s_max = self.s + self.drift_max
        else:
            raise ValueError(f"invalid action name: {a_name}")
        s_min_fixed = max(0, s_min)
        s_max_fixed = min(self.s_max_val, s_max) 
        assert s_min_fixed < s_max_fixed
        return s_min_fixed, s_max_fixed

    def _transition_regular(self, action):
        s_min, s_max = self._get_transition_min_max(action)
        return np.random.uniform(s_min, s_max)

    def _transition_adversarial(self, action):
        s_min_raw, s_max = self._get_transition_min_max(action)
        adversarial_alpha = 1 / (1 + self.adversarial_lambda)
        s_min = s_max - adversarial_alpha * (s_max - s_min_raw)
        return np.random.uniform(s_min, s_max)
 
    def get_s_dim(self):
        return 1

    def get_num_a(self):
        return 2

    def get_s_a_init(self, pi_e, device=None):
        s_init = np.array([self.s_init], dtype="float32")
        a_init = pi_e(s_init)
        s_init_torch = torch.from_numpy(s_init)
        a_init_torch = torch.LongTensor([a_init])
        if device is None:
            return s_init_torch, a_init_torch
        else:
            return s_init_torch.to(device), a_init_torch.to(device)

    def flexible_basis_func(self, s, a=None):
        if a is None:
            a = torch.zeros(len(s)).long().to(s.device)
        s_max = self.s_max_val
        step = 1.0 * s_max / self.num_init_basis_func
        min_thresholds = torch.arange(
            start=0, end=s_max, step=step
        ).to(s.device).unsqueeze(0)
        # max_thresholds = min_thresholds + step
        # f_1 = (s >= min_thresholds) * (s <= max_thresholds) * 1.0
        f_1 = (s <= min_thresholds) * 1.0
        f_2 = F.one_hot(a, num_classes=2)
        bias = torch.ones_like(s)
        f = (f_1.unsqueeze(1) * f_2.unsqueeze(2)).reshape(len(s), -1)
        return torch.cat([bias, f], dim=1)

    def get_num_init_basis_func(self):
        return 2 * self.num_init_basis_func + 1

    def bias_basis_func(self, s, a=None):
        return torch.ones_like(s)