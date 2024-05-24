
import torch
import util
from ipdb import set_trace as debug

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def sample_e(opt, x, num_samples=1):
    sample_func = {'gaussian': sample_gaussian_like, 'rademacher': sample_rademacher_like,
        }.get(opt.noise_type)

    if num_samples == 1:
        return sample_func(x)
    else:
        return [sample_func(x) for _ in range(num_samples)]

def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):

    zs = policy(xs,ts)
    g_ts = dyn.g(ts)

    # g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    if opt.problem_name in ['mnist','cifar10','celebA32','celebA64']:
        g_ts = g_ts[:,None,None,None]  # (B) (B,1,1,1)
    elif opt.problem_name in ['gmm', 'checkerboard', 'moon-to-spiral', 'moon', 'spiral',
        'gaussian-to-gaussian', 'inception']:
        g_ts = g_ts[:,None]  # (B) (B,1)
    elif opt.problem_name in ['sinusoid', 'sinusoid_large', 'pm25', 'physio',
        'tba2017', 'etf052023',
        'exchange_rate_nips', 'solar_nips', 'electricity_nips', ]:
        g_ts = g_ts[:,None,None,None]  # (B) (B,1,1,1)
    else:
        raise NotImplementedError('New dataset.')

    gzs = g_ts * zs

    # From torch doc: create_graph if True, graph of the derivative will be constructed, allowing
    # to compute higher order derivative products. As the target equation involves the gradient,
    # so we need to compute the gradient (over model parameters) of gradient (over data x).
    if opt.num_hutchinson_samp == 1:
        e = sample_e(opt, xs)
        e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
        div_gz = e_dzdx * e
        # approx_div_gz = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    else:
        # Unit test see: Hutchinson-Test-Sinusoid.ipynb
        div_gz = 0
        for hut_id in range(opt.num_hutchinson_samp):
            e = sample_e(opt, xs)
            e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True, retain_graph=True)[0]
            div_gz += e_dzdx * e
        div_gz = div_gz / opt.num_hutchinson_samp

    return [div_gz, zs] if return_zs else div_gz


def compute_div_gz_imputation(
        opt, dyn, ts, xs,
        obs_data, obs_mask, cond_mask, gt_mask,
        policy, return_zs=False):
    assert policy.direction == 'backward'

    if getattr(opt, policy.direction + '_net') in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
        cond_obs =  obs_data * cond_mask
        noisy_target = (1-cond_mask) * xs  # no big difference if using  target_mask * x
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        diff_input = (total_input, cond_mask)

    elif getattr(opt, policy.direction + '_net') == 'Transformerv3':
        cond_obs = cond_mask * obs_data
        noisy_target = (1-cond_mask) * xs
        total_input = cond_obs + noisy_target
        diff_input = (total_input, cond_mask)

    else:
        diff_input = xs

    zs = policy(diff_input, ts)
    g_ts = dyn.g(ts)

    # g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    if opt.problem_name in ['mnist','cifar10','celebA32','celebA64']:
        g_ts = g_ts[:,None,None,None]  # (B) (B,1,1,1)
    elif opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral', 'gaussian-to-gaussian', 'moon', 'spiral']:
        g_ts = g_ts[:,None]  # (B) (B,1)
    elif opt.problem_name in ['sinusoid', 'sinusoid_large', 'pm25', 'physio',
        'tba2017', 'etf052023',
        'exchange_rate_nips', 'solar_nips', 'electricity_nips', ]:
        g_ts = g_ts[:,None,None,None]  # (B) (B,1,1,1)
    else:
        raise NotImplementedError('New dataset.')

    gzs = g_ts * zs

    # From torch doc: create_graph if True, graph of the derivative will be constructed, allowing
    # to compute higher order derivative products. As the target equation involves the gradient,
    # so we need to compute the gradient (over model parameters) of gradient (over data x).
    if opt.num_hutchinson_samp == 1:
        e = sample_e(opt, xs)
        e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
        div_gz = e_dzdx * e
        # approx_div_gz = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    else:
        # Unit test see: Hutchinson-Test-Sinusoid.ipynb
        div_gz = 0
        for hut_id in range(opt.num_hutchinson_samp):
            e = sample_e(opt, xs)
            e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True, retain_graph=True)[0]
            div_gz += e_dzdx * e
        div_gz = div_gz / opt.num_hutchinson_samp

    return [div_gz, zs] if return_zs else div_gz


def compute_sb_nll_alternate_train(opt, dyn, ts, xs, zs_impt, policy_opt, return_z=False):
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert opt.train_method in ['alternate', 'alternate_backward',
        'alternate_backward_imputation', 'alternate_backward_imputation_v2',
        'alternate_imputation', 'alternate_imputation_v2']
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(opt, dyn, ts, xs, policy_opt, return_zs=True)
        loss = zs*(0.5*zs + zs_impt) + div_gz
        # print('div_gz', torch.sum(div_gz * dyn.dt) / batch_x / batch_t)
        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
    return loss, zs if return_z else loss


def compute_sb_nll_alternate_imputation_train(
    opt, dyn, ts, xs, zs_impt,
    obs_data, obs_mask, cond_mask, gt_mask,
    policy_opt, return_z=False):
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert opt.train_method in ['alternate', 'alternate_backward',
        'alternate_backward_imputation', 'alternate_backward_imputation_v2',
        'alternate_imputation', 'alternate_imputation_v2']
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz_imputation(opt, dyn, ts, xs,
            obs_data, obs_mask, cond_mask, gt_mask,
            policy_opt, return_zs=True)

        if opt.backward_net in ['Transformerv2', 'Transformerv3', 'Transformerv4', 'Transformerv5']:
            loss_mask = obs_mask - cond_mask
        else:
            loss_mask = obs_mask

        zs  = zs * loss_mask
        zs_impt = zs_impt * loss_mask
        div_gz = div_gz * loss_mask
        loss = zs*(0.5*zs + zs_impt) + div_gz
        # print('div_gz', torch.sum(div_gz * dyn.dt) / batch_x / batch_t)
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

