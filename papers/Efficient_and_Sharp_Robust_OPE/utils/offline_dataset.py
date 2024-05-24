import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from policies.abstract_policy import AbstractPolicy

class OfflineRLDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.s = torch.FloatTensor([])
        self.a = torch.LongTensor([])
        self.ss = torch.FloatTensor([])
        self.r = torch.FloatTensor([])
        self.pi_s = {}
        self.pi_ss = {}
        self.pi_b_a_probs = None

    def sample_new_trajectory(self, env, pi, burn_in, num_sample, thin=1):
        assert isinstance(pi, AbstractPolicy)

        # first, run some burn-in iterations
        s, _ = env.reset()
        for _ in range(burn_in):
            a = pi(s)
            s, _, _, _, _ = env.step(a)

        # now, sample data from stationary distribution
        s_list = []
        ss_list = []
        a_list = []
        r_list = []
        for i_ in range(num_sample * thin):
            a = pi(s)
            ss, r, _, _, _ = env.step(a)
            if i_ % thin == 0:
                s_list.append(torch.from_numpy(s))
                a_list.append(a)
                ss_list.append(torch.from_numpy(ss))
                r_list.append(r)
            s = ss

        # finally, convert sampled data into tensors
        self.s = torch.cat([self.s, torch.stack(s_list)])
        self.a = torch.cat([self.a, torch.LongTensor(a_list)])
        self.ss = torch.cat([self.ss, torch.stack(ss_list)])
        self.r = torch.cat([self.r, torch.FloatTensor(r_list)])

    def get_batch_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size)

    def apply_eval_policy(self, pi_eval_name, pi_eval):
        assert pi_eval_name not in self.pi_ss

        pi_s_list = [pi_eval(s_.cpu().numpy()) for s_ in self.s]
        self.pi_s[pi_eval_name] = torch.LongTensor(pi_s_list)

        pi_ss_list = [pi_eval(ss_.cpu().numpy()) for ss_ in self.ss]
        self.pi_ss[pi_eval_name] = torch.LongTensor(pi_ss_list)

    def compute_pi_b_probs(self, pi_b_base, epsilon):
        base_a = torch.LongTensor([pi_b_base(s_.cpu().numpy()) for s_ in self.s])
        base_a = base_a.to(self.a.device)
        num_a = len(set([int(a_) for a_ in self.a]))
        self.pi_b_a_probs = epsilon / num_a + (1 - epsilon) * (self.a == base_a)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        x = {
            "s": self.s[i],
            "a": self.a[i],
            "ss": self.ss[i],
            "r": self.r[i],
        }
        for pi_eval_name, pi_ss in self.pi_ss.items():
            x[f"pi_ss::{pi_eval_name}"] = pi_ss[i]
            pi_s = self.pi_s[pi_eval_name]
            x[f"pi_s::{pi_eval_name}"] = pi_s[i]
        if self.pi_b_a_probs is not None:
            x["pi_b_probs"] = self.pi_b_a_probs[i]
        return x

    def get_train_dev_split(self, dev_frac):
        num_dev = int(len(self.s) * dev_frac)
        num_train = len(self.s) - num_dev
        train_split = self.get_split_dataset(0, num_train)
        dev_split = self.get_split_dataset(num_train, num_train+num_dev)
        return train_split, dev_split

    def get_split_dataset(self, start_i, end_i):
        split = OfflineRLDataset()
        split.s = self.s[start_i:end_i]
        split.a = self.a[start_i:end_i]
        split.ss = self.ss[start_i:end_i]
        split.r = self.r[start_i:end_i]
        for pi_name, pi_ss in self.pi_ss.items():
            split.pi_ss[pi_name] = pi_ss[start_i:end_i]
            pi_s = self.pi_s[pi_name]
            split.pi_s[pi_name] = pi_s[start_i:end_i]
            split.pi_ss[pi_name] = pi_ss[start_i:end_i]
        if self.pi_b_a_probs is not None:
            split.pi_b_a_probs = self.pi_b_a_probs[start_i:end_i]
        return split

    def save_dataset(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.s, os.path.join(save_dir, "s.pt"))
        torch.save(self.a, os.path.join(save_dir, "a.pt"))
        torch.save(self.ss, os.path.join(save_dir, "ss.pt"))
        torch.save(self.r, os.path.join(save_dir, "r.pt"))
        pi_e_name_list = list(self.pi_s)
        with open(os.path.join(save_dir, "pi_names.json"), "w") as f:
            json.dump(pi_e_name_list, f)
        for pi_name, pi_ss in self.pi_ss.items():
            pi_s = self.pi_s[pi_name]
            torch.save(pi_s, os.path.join(save_dir, f"{pi_name}__pi_s.pt"))
            torch.save(pi_ss, os.path.join(save_dir, f"{pi_name}__pi_ss.pt"))
        if self.pi_b_a_probs is not None:
            pi_b_a_path = os.path.join(save_dir, "pi_b_a_probs.pt")
            torch.save(self.pi_b_a_probs, pi_b_a_path)

    @staticmethod
    def load_dataset(load_dir):
        dataset = OfflineRLDataset()
        dataset.s = torch.load(os.path.join(load_dir, "s.pt"))
        dataset.a = torch.load(os.path.join(load_dir, "a.pt"))
        dataset.ss = torch.load(os.path.join(load_dir, "ss.pt"))
        dataset.r = torch.load(os.path.join(load_dir, "r.pt"))
        with open(os.path.join(load_dir, "pi_names.json")) as f:
            pi_name_list = json.load(f)
        for pi_name in pi_name_list:
            pi_s_path = os.path.join(load_dir, f"{pi_name}__pi_s.pt")
            pi_ss_path = os.path.join(load_dir, f"{pi_name}__pi_ss.pt")
            dataset.pi_s[pi_name] = torch.load(pi_s_path)
            dataset.pi_ss[pi_name] = torch.load(pi_ss_path)
        pi_b_a_path = os.path.join(load_dir, "pi_b_a_probs.pt")
        if os.path.exists(pi_b_a_path):
            dataset.pi_b_a_probs = torch.load(pi_b_a_path)
        return dataset

    def to(self, device):
        if device is None:
            return
        self.s = self.s.to(device)
        self.a = self.a.to(device)
        self.ss = self.ss.to(device)
        self.r = self.r.to(device)
        for pi_name in self.pi_s.keys():
            self.pi_s[pi_name] = self.pi_s[pi_name].to(device)
            self.pi_ss[pi_name] = self.pi_ss[pi_name].to(device)
        if self.pi_b_a_probs is not None:
            self.pi_b_a_probs = self.pi_b_a_probs.to(device)
