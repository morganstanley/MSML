from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from models.abstract_nuisance_model import AbstractNuisanceModel

class AbstractLearner(ABC):
    def __init__(self, nuisance_model, gamma, adversarial_lambda,
                 train_q_beta=False, train_eta=True, train_w=False,
                 worst_case=True, use_dual_cvar=True):
        self.model = nuisance_model
        self.prev_model = None
        self.gamma = gamma
        self.adversarial_lambda = adversarial_lambda
        self.worst_case = worst_case
        self.train_q_beta = train_q_beta
        self.train_eta = train_eta
        self.train_w = train_w
        self.use_dual_cvar = use_dual_cvar
        assert isinstance(self.model, AbstractNuisanceModel)
        super().__init__()

    def update_prev_model(self):
        self.prev_model = self.model.get_copy()
        self.prev_model.eval()

    def get_num_moments(self):
        # return 2 * self.train_q_beta + 1 * self.train_eta + 1 * self.train_w
        return 1 * self.train_q_beta + 1 * self.train_eta + 1 * self.train_w

    def get_batch_quantile_loss(self, batch, pi_e_name):
        lmbda = self.adversarial_lambda
        alpha = 1 / (1 + lmbda)

        s = batch["s"]
        # s_noise = torch.randn(s.shape).to(s.device) * 0
        # s = s + s_noise
        a = batch["a"]
        ss = batch["ss"]
        pi_ss = batch[f"pi_ss::{pi_e_name}"]

        v, beta = self.model.get_v_beta(s, a, ss, pi_ss)
        if self.worst_case:
            loss = (1 - alpha) * F.relu(beta - v) + alpha * F.relu(v - beta)
        else:
            loss = (1 - alpha) * F.relu(v - beta) + alpha * F.relu(beta - v)
        # multiply by 1 - self.gamma so loss is on gamma-invariant scale
        return (1 - self.gamma) * loss.mean()
        # return 100.0 * loss.mean()

    def get_batch_q_target(self, s, a, ss, r, pi_ss, pi_e_name, use_prev_model=False):
        if use_prev_model:
            model = self.prev_model
        else:
            model = self.model
        v, beta = model.get_v_beta(s, a, ss, pi_ss)

        lmbda = self.adversarial_lambda
        inv_lmbda = lmbda ** -1
        assert inv_lmbda > 0
        assert inv_lmbda <= 1
        if self.use_dual_cvar:
            if self.worst_case:
                cvar_v = beta - (1 + lmbda) * F.relu(beta - v)
            else:
                cvar_v = beta + (1 + lmbda) * F.relu(v - beta)
        else:
            if self.worst_case:
                cvar_v = (1 + lmbda) * (beta > v) * v
            else:
                cvar_v = (1 + lmbda) * (v > beta) * v
        e_cvar_v = inv_lmbda * v + (1 - inv_lmbda) * cvar_v
        return r.unsqueeze(-1) + self.gamma * e_cvar_v

    def get_batch_q_loss(self, batch, pi_e_name, detach_target=True):
        s = batch["s"]
        a = batch["a"]
        ss = batch["ss"]
        r = batch["r"]
        pi_ss = batch[f"pi_ss::{pi_e_name}"]

        q = self.model.get_q(s, a)
        q_target = self.get_batch_q_target(s, a, ss, r, pi_ss, pi_e_name,
                                           use_prev_model=True)
        if detach_target:
            q_target = q_target.detach()
        q_err = (1 - self.gamma) * (q - q_target)
        return (q_err ** 2).mean()

    def get_critic_basis_expansion(self, batch, critic):
        basis_list = []
        if self.train_q_beta:
            q_basis = critic.get_q_basis_expansion(
                s=batch["s"], a=batch["a"]
            )
            basis_list.append(q_basis)
            # basis_list.append(beta_basis)
        if self.train_eta:
            eta_basis = critic.get_eta_basis_expansion(
                s=batch["s"], a=batch["a"]
            )
            basis_list.append(eta_basis)
        if self.train_w:
            w_basis = critic.get_w_basis_expansion(
                s=batch["s"]
            )
            basis_list.append(w_basis)
        f_basis = torch.stack(basis_list, dim=2).detach()
        return f_basis

    def get_batch_moments(self, batch, critic, pi_e_name, s_init,
                          model_grad=False, critic_grad=False,
                          basis_expansion=False):
        s = batch["s"]
        a = batch["a"]
        ss = batch["ss"]
        r = batch["r"]
        pi_s = batch[f"pi_s::{pi_e_name}"]
        pi_ss = batch[f"pi_ss::{pi_e_name}"]

        moments_list = []
        if self.train_q_beta:
            q_m = self.get_q_batch_moments(
                critic=critic, s=s, a=a, ss=ss, r=r, pi_ss=pi_ss,
                model_grad=model_grad, critic_grad=critic_grad,
                basis_expansion=basis_expansion
            )
            moments_list.append(q_m)

        if self.train_eta:
            eta_m = self.get_eta_batch_moments(
                critic=critic, s=s, a=a, pi_s=pi_s,
                model_grad=model_grad, critic_grad=critic_grad,
                basis_expansion=basis_expansion
            )
            moments_list.append(eta_m)

        if self.train_w:
            w_m = self.get_w_batch_moments(
                critic=critic, s=s, a=a, ss=ss, pi_s=pi_s, pi_ss=pi_ss,
                s_init=s_init, model_grad=model_grad, critic_grad=critic_grad,
                basis_expansion=basis_expansion
            )
            moments_list.append(w_m)


        if basis_expansion:
            return torch.stack(moments_list, dim=2)
        else:
            return torch.cat(moments_list, dim=1).sum(1)

    def get_batch_l2_reg_model(self, batch, pi_e_name):
        s = batch["s"]
        a = batch["a"]
        ss = batch["ss"]
        pi_ss = batch[f"pi_ss::{pi_e_name}"]
        q, v, _, eta, w  = self.model.get_all(s, a, ss, pi_ss)
        reg_sum = 0
        if self.train_q_beta:
            reg_sum += (q ** 2).mean()
            reg_sum += (v ** 2).mean()
        if self.train_eta:
            reg_sum += (eta ** 2).mean()
        if self.train_w:
            reg_sum += (w ** 2).mean()
        return reg_sum

    def get_batch_l2_reg_beta(self, batch):
        s = batch["s"]
        a = batch["a"]
        beta = self.model.get_beta(s, a)
        return (beta ** 2).mean()

    def get_batch_l2_reg_critic(self, batch, critic):
        s = batch["s"]
        a = batch["a"]
        f_q, f_eta, f_w  = critic.get_all(s, a)
        reg_sum = 0
        if self.train_q_beta:
            reg_sum += (f_q ** 2).mean()
        if self.train_eta:
            reg_sum += (f_eta ** 2).mean()
        if self.train_w:
            reg_sum += (f_w ** 2).mean()
        return reg_sum

    def get_q_batch_moments(self, critic, s, a, ss, r, pi_ss,
                            model_grad=False, critic_grad=False,
                            basis_expansion=False):


        q = self.model.get_q(s, a)
        q_target = self.get_batch_q_target(s, a, ss, r, pi_ss, pi_e_name)
        rho_q = (1 - self.gamma) * (q - q_target)
        if not model_grad:
            rho_q = rho_q.detach()

        if basis_expansion:
            f_q = critic.get_q_basis_expansion(s, a)
        else:
            f_q = critic.get_q(s, a)

        if not critic_grad:
            f_q = f_q.detach()

        return rho_q * f_q

    def get_eta_batch_moments(self, critic, s, a, pi_s,
                              model_grad=False, critic_grad=False,
                              basis_expansion=False):
        pi_e_match = (pi_s == a).reshape(-1, 1) * 1.0
        eta = self.model.get_eta(s, a) * pi_e_match
        if not model_grad:
            eta = eta.detach()
        if basis_expansion:
            f_eta_s_a = critic.get_eta_basis_expansion(s, a)
            f_eta_s_pi = critic.get_eta_basis_expansion(s, pi_s)
        else:
            f_eta_s_a = critic.get_eta(s, a)
            f_eta_s_pi = critic.get_eta(s, pi_s)
        if not critic_grad:
            f_eta_s_a = f_eta_s_a.detach()
            f_eta_s_pi = f_eta_s_pi.detach()

        return f_eta_s_a * eta - f_eta_s_pi

    def get_w_batch_moments(self, critic, s, a, ss, pi_s, pi_ss, s_init,
                            model_grad=False, critic_grad=False,
                            basis_expansion=False):
        w = self.model.get_w(s)
        w_ss = self.model.get_w(ss)
        pi_e_match = (pi_s == a).reshape(-1, 1) * 1.0
        eta = self.model.get_eta(s, a) * pi_e_match
        v, beta = self.model.get_v_beta(s, a, ss, pi_ss)
        if self.worst_case:
            xi = (beta > v) * 1.0
        else:
            xi = (v > beta) * 1.0
        # xi = self.model.get_xi(s, a, ss, pi_ss)
        if not model_grad:
            w = w.detach()
            w_ss = w_ss.detach()
            eta = eta.detach()
            xi = xi.detach()
        if not self.train_q_beta:
            xi = xi.detach()
        if not self.train_eta:
            eta = eta.detach()

        if basis_expansion:
            f_s = critic.get_w_basis_expansion(s)
            f_ss = critic.get_w_basis_expansion(ss)
            f_s0 = critic.get_w_basis_expansion(s_init.unsqueeze(0))
        else:
            f_s = critic.get_w(s)
            f_ss = critic.get_w(ss)
            f_s0 = critic.get_w(s_init.unsqueeze(0))
        if not critic_grad:
            f_s = f_s.detach()
            f_ss = f_ss.detach()
            f_s0 = f_s0.detach()

        lmbda = self.adversarial_lambda
        inv_lmbda = lmbda ** -1
        lambda_is = inv_lmbda + (1 - inv_lmbda) * (1 + lmbda) * xi
        # return (self.gamma * w * (eta * lambda_is).detach() * f_ss
        #         - lambda_is.detach() * f_ss * w_ss + (1 - self.gamma) * f_s0)
        return (self.gamma * w * (eta * lambda_is) * f_ss
                - f_ss * w_ss + (1 - self.gamma) * f_s0)

    def print_eta_error_info(self, dl, critic, pi_e_name, batch_scale=1000.0):
        est_err_sum = 0.0
        true_err_sum = 0.0
        eta_err_sum = 0.0
        batch_size_sum = 0

        for batch in dl:
            s = batch["s"]
            a = batch["a"]
            ss = batch["ss"]
            pi_s = batch[f"pi_s::{pi_e_name}"]
            pi_b_probs = batch["pi_b_probs"].reshape(-1, 1)
            pi_e_match = (pi_s == a).reshape(-1, 1) * 1.0

            f_eta_s_a = critic.get_eta_basis_expansion(s, a)
            f_eta_s_pi = critic.get_eta_basis_expansion(s, pi_s)
            eta = self.model.get_eta(s, a) * pi_e_match
            true_eta = pi_e_match / pi_b_probs

            est_err_sum += (f_eta_s_a * eta - f_eta_s_pi).sum(0) / batch_scale
            true_err_sum += (f_eta_s_a * true_eta - f_eta_s_pi).sum(0) / batch_scale
            eta_err_sum += ((eta - true_eta) ** 2).sum() / batch_scale
            batch_size_sum += len(batch["s"]) / batch_scale

        est_err = est_err_sum / batch_size_sum
        true_err = true_err_sum / batch_size_sum
        eta_err = eta_err_sum / batch_size_sum
        print(f"max moment error with estimated eta: {est_err.abs().max()}")
        print(f"max moment error with true eta: {true_err.abs().max()}")
        print(f"L2 error of estimated eta: {eta_err}")

    @abstractmethod
    def train(self, dataset, pi_e_name):
        pass