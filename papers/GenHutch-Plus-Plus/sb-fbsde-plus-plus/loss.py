
import torch
import util
from ipdb import set_trace as debug
from matrix_utils import vjp, jvp

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def sample_e(opt, x):
    return {
        'gaussian': sample_gaussian_like,
        'rademacher': sample_rademacher_like,
    }.get(opt.noise_type)(x)


def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):

    zs = policy(xs,ts)

    g_ts = dyn.g(ts)
    g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    gzs = g_ts*zs

    if opt.div_method == 'h':

        e = sample_e(opt, xs)
        e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
        div_gz = e_dzdx * e
        # approx_div_gz = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    elif opt.div_method == 'hpp':
        S = sample_e(opt, xs)  # [bs, 1, K, L]
        G = sample_e(opt, xs)  # [bs, 1, K, L]
        if opt.div_accelerate:
            # only used the first and the last sample
            # first_t_indices = torch.arange(0, opt.train_bs_x * opt.train_bs_t, opt.train_bs_t)
            last_t_indices = torch.arange(opt.train_bs_t - 1, opt.train_bs_x * opt.train_bs_t, opt.train_bs_t)
            # t_indices = torch.cat([first_t_indices, last_t_indices])


            dzdx_S = jvp(gzs[last_t_indices], xs[last_t_indices], S[last_t_indices], create_graph=True)  # [1 * opt.train_bs_x, 1, K, L]
            dzdx_S = dzdx_S.squeeze(1).unsqueeze(-1)  # [1 * opt.train_bs_x, K, L, 1]
            dzdx_S = dzdx_S.contiguous().view(dzdx_S.shape[0], -1, 1)  # [1 * opt.train_bs_x, K * L, 1]

            [Q_tmp, _] = torch.linalg.qr(dzdx_S, mode='reduced')  # [1 * opt.train_bs_x, K * L, 1]
            Q = torch.zeros((xs.shape[0], xs.shape[-2]*xs.shape[-1], 1))  # (bs_x, bs_t, K*L, 1)
            Q.view(opt.train_bs_x, opt.train_bs_t, xs.shape[-2]*xs.shape[-1], 1)[:, :-1] = Q_tmp.unsqueeze(1)

            Q = Q.view(opt.train_bs_x * opt.train_bs_t, xs.shape[-2]*xs.shape[-1], 1)
        else:
            dzdx_S = jvp(gzs, xs, S, create_graph=True)  # [bs, 1, K, L]
            dzdx_S = dzdx_S.squeeze(1).unsqueeze(-1)  # [bs, K, L, 1]
            dzdx_S = dzdx_S.contiguous().view(xs.shape[0], -1, 1)  # [bs, C * K * L, 1]

            [Q, _] = torch.linalg.qr(dzdx_S, mode='reduced')  # [bs, C * K * L, 1]

        # update G = G - Q @ (Q^T @ G)
        # Chenck detach() ?
        Q = Q.detach()
        # Q = Q.view(xs.shape[0], xs.shape[-2], xs.shape[-1], 1).unsqueeze(1)  # [bs, 1, K, L, 1]
        G = G.squeeze(1).unsqueeze(-1).contiguous().view(xs.shape[0], -1, 1)  
        G = G - torch.bmm(Q, torch.bmm(Q.permute(0, 2, 1), G))  # [bs, C * K * L, 1]

        Q = Q.contiguous().view(xs.shape[0], xs.shape[-3], xs.shape[-2], xs.shape[-1], 1).squeeze(-1)  # [bs, C, K, L]
        G = G.contiguous().view(xs.shape[0], xs.shape[-3], xs.shape[-2], xs.shape[-1], 1).squeeze(-1)  # [bs, C, K, L]
        # compute tr(Q^T @ A @ Q)
        Q_dzdx = torch.autograd.grad(gzs, xs, Q, create_graph=True)[0]
        Q_dzdx_Q = Q_dzdx * Q

        # compute tr(G^T @ A @ G)       
        # hutchinson's estimator == divergence_approx(f, y, e=G)
        G_dzdx = torch.autograd.grad(gzs, xs, G, create_graph=True)[0]
        G_dzdx_G = G_dzdx * G

        div_gz = Q_dzdx_Q + G_dzdx_G


    return [div_gz, zs] if return_zs else div_gz


def compute_sb_nll_alternate_train(opt, dyn, ts, xs, zs_impt, policy_opt, return_z=False):
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert opt.train_method == 'alternate'
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(opt, dyn, ts, xs, policy_opt, return_zs=True)
        loss = zs*(0.5*zs + zs_impt) + div_gz
        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
    return loss, zs if return_z else loss


def compute_sb_nll_joint_train(opt, batch_x, dyn, ts, xs_f, zs_f, x_term_f, policy_b):
    """ Implementation of Eq (16) in our main paper.
    """
    assert opt.train_method == 'joint'
    assert policy_b.direction == 'backward'
    assert xs_f.requires_grad and zs_f.requires_grad and x_term_f.requires_grad

    div_gz_b, zs_b = compute_div_gz(opt, dyn, ts, xs_f, policy_b, return_zs=True)

    loss = 0.5*(zs_f + zs_b)**2 + div_gz_b
    loss = torch.sum(loss*dyn.dt) / batch_x
    loss = loss - dyn.q.log_prob(x_term_f).mean()
    return loss

