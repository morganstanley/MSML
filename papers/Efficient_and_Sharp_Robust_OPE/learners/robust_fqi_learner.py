from copy import deepcopy
import sys

import torch
from torch.optim import Adam, LBFGS
import torch.nn as nn
import numpy as np

from learners.abstract_learner import AbstractLearner
from models.abstract_critic import AbstractCritic
from utils.oadam import OAdam


class RobustFQILearner(AbstractLearner):
    def __init__(self, nuisance_model, gamma, adversarial_lambda,
                 use_dual_cvar=True, worst_case=True):
        super().__init__(
            nuisance_model=nuisance_model, gamma=gamma,
            adversarial_lambda=adversarial_lambda, worst_case=worst_case,
            use_dual_cvar=use_dual_cvar,
            train_q_beta=True, train_eta=False, train_w=False
        )

    def train(self, dataset, pi_e_name, evaluate_pv_kwargs=None, batch_size=1024,
              max_num_iterations=100, num_restart=3,
              beta_reg_alpha=1e-5, q_reg_alpha=1e-5, min_q_change=1e-5,
              verbose=False, device=None):

        # beta_data, q_data = dataset.get_train_dev_split(0.5)
        # dl_beta = beta_data.get_batch_loader(batch_size)
        # dl_q = q_data.get_batch_loader(batch_size)
        dl = dataset.get_batch_loader(batch_size)
        dl_beta = dl
        dl_q = dl

        min_err = float("inf")
        best_state = None

        for restart_i in range(num_restart):
            try:
                self.model.reset_networks()
                next_min_err, next_best_state = self._try_single_train(
                    dl_beta=dl_beta, dl_q=dl_q, pi_e_name=pi_e_name,
                    evaluate_pv_kwargs=evaluate_pv_kwargs, batch_size=batch_size,
                    max_num_iterations=max_num_iterations, num_restart=num_restart,
                    beta_reg_alpha=beta_reg_alpha, q_reg_alpha=q_reg_alpha,
                    min_q_change=min_q_change, verbose=verbose, device=device,
                )
                if next_min_err < min_err:
                    min_err = next_min_err
                    best_state = next_best_state
            except Exception as e:
                print(f"WARNING: exception occurred in trying to train q/beta model")
                if hasattr(e, "message"):
                    print(e.message)
                else:
                    print(e)


        self.model.set_state(best_state)
        self.model.eval()

    def _try_single_train(self, dl_beta, dl_q, pi_e_name, evaluate_pv_kwargs=None,
                batch_size=1024, max_num_iterations=100, num_restart=3,
                beta_reg_alpha=1e-5, q_reg_alpha=1e-5, min_q_change=1e-5,
                verbose=False, device=None):
        min_err = float("inf")
        best_state = None

        # do an initial update on beta from starting q
        self.model.train()
        beta_optim = LBFGS(self.model.get_beta_parameters(),
                            line_search_fn="strong_wolfe")
        self.train_func_one_epoch(
            lbfgs_optim=beta_optim, dl=dl_beta, pi_e_name=pi_e_name,
            beta_update=True, alpha_reg=beta_reg_alpha, 
        )
        self.model.eval()
        self.update_prev_model()

        # now do iterative updates with weighted objective
        for iter_i in range(1, max_num_iterations+1):
            if verbose:
                print("")
                print(f"starting iteration {iter_i}")

            # do a training update on q
            self.model.train()
            q_optim = LBFGS(self.model.get_parameters())
            self.train_func_one_epoch(
                lbfgs_optim=q_optim, dl=dl_q, pi_e_name=pi_e_name,
                beta_update=False, alpha_reg=q_reg_alpha, 
            )
            self.model.eval()

            q_l2_change = self.compute_q_l2_change(dl_q, pi_e_name)
            # if change from previous model is sufficiently small then stop early
            if verbose:
                print(f"Q function L2 change: {q_l2_change}")
            if q_l2_change < min_q_change:
                break

            # do a training update on beta
            self.model.train()
            beta_optim = LBFGS(self.model.get_beta_parameters(),
                            line_search_fn="strong_wolfe")
            self.train_func_one_epoch(
                lbfgs_optim=beta_optim, dl=dl_beta, pi_e_name=pi_e_name,
                beta_update=True, alpha_reg=beta_reg_alpha, 
            )
            self.model.eval()
            self.update_prev_model()

            q_err = self.compute_q_error(dl_q, pi_e_name)
            if verbose:
                print(f"robust Q error: {q_err}")

            if q_err < min_err:
                min_err = q_err
                best_state = deepcopy(self.model.get_state())
                if verbose:
                    print("NEW BEST")

            if verbose and (evaluate_pv_kwargs is not None):
                self.print_policy_value_estimate(**evaluate_pv_kwargs)

        # self.model.set_state(best_state)
        # self.model.eval()
        return min_err, best_state

    def print_policy_value_estimate(self, s_init, a_init, pi_e_name, dl_test):
        q_pv = self.model.estimate_policy_val_q(
            s_init=s_init, a_init=a_init, gamma=self.gamma
        )
        print(f"Intermediate policy value results:")
        print(f"Q-estimated v(pi_e): {q_pv}")

    def train_func_one_epoch(self, lbfgs_optim, dl, pi_e_name, alpha_reg,
                             beta_update=False, batch_scale=1000.0):
        def closure():
            lbfgs_optim.zero_grad()
            loss_sum = 0
            batch_sum = 0
            for batch in dl:
                if beta_update:
                    fit_loss = self.get_batch_quantile_loss(
                        batch=batch, pi_e_name=pi_e_name,
                    )
                else:
                    fit_loss = self.get_batch_q_loss(
                        batch=batch, pi_e_name=pi_e_name,
                    )
                if alpha_reg and beta_update:
                    reg = alpha_reg * self.get_batch_l2_reg_beta(batch)
                elif alpha_reg and (not beta_update):
                    reg = alpha_reg * self.get_batch_l2_reg_model(batch, pi_e_name)
                else:
                    reg = 0
                loss = fit_loss + reg
                batch_weight = len(batch["s"]) / batch_scale
                loss_sum += batch_weight * loss
                batch_sum += batch_weight
            total_loss = loss_sum / batch_sum
            total_loss.backward()
            return total_loss

        lbfgs_optim.step(closure)

    def compute_q_error(self, dl, pi_e_name, batch_scale=1000.0):
        squared_err_sum = 0
        batch_sum = 0
        for batch in dl:
            batch_mse = float(self.get_batch_q_loss(batch, pi_e_name))
            batch_weight = len(batch["s"]) / batch_scale
            squared_err_sum += batch_weight * batch_mse
            batch_sum += batch_weight
        return squared_err_sum / batch_sum

    def compute_q_l2_change(self, dl, pi_e_name, batch_scale=1000.0):
        squared_err_sum = 0
        batch_sum = 0
        for batch in dl:
            s = batch["s"]
            a = batch["a"]
            q_square_err = (self.model.get_q(s, a) - self.prev_model.get_q(s, a)) ** 2
            batch_weight = len(batch["s"]) / batch_scale
            squared_err_sum += batch_weight * float(q_square_err.mean())
            batch_sum += batch_weight
        return squared_err_sum / batch_sum