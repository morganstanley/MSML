import numpy as np


def evaluate_policy(env, policy, gamma, min_prec=1e-3, min_num_rollout=1, max_num_rollout=1e9):
    r_list = []
    num_rollout = 1000
    s, _ = env.reset()
    while True:
        for _ in range(num_rollout):
            a = policy(s)
            s_next, r, _, _, _ = env.step(a)
            r_list.append(r)
            # print(s, a, s_next, r)
            if np.random.randn() > gamma:
                # print("RESET")
                s, _ = env.reset()
            else:
                s = s_next
        r_std = np.std(r_list)
        # print(len(r_list), np.mean(r_list), r_std)
        if len(r_list) >= min_num_rollout and r_std / (len(r_list) ** 0.5) <= min_prec:
            break
        elif len(r_list) > max_num_rollout:
            break
        else:
            num_rollout = len(r_list)
    return np.mean(r_list)
            