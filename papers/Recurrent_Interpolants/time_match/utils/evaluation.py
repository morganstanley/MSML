import numpy as np
import torch
import ot as pot
import math


def wasserstein(
    gen_samples: torch.Tensor,
    target: torch.Tensor,
) -> float:
    a, b = pot.unif(gen_samples.shape[0]), pot.unif(target.shape[0])
    M = torch.cdist(gen_samples, target)
    M = M**2
    ret = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    return math.sqrt(ret)


def mmd(
    gen_samples: torch.Tensor,
    target: torch.Tensor,
    kernel='multiscale',
    device='cpu'
) -> float:
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Ref: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf?ref=https://githubhelp.com
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. The Journal of Machine Learning Research, 13(1), 723-773.

    https://arxiv.org/pdf/1806.07755.pdf
    An empirical study on evaluation metrics of generative adversarial networks. Qiantong Xu
    """
    xx = torch.mm(gen_samples, gen_samples.t())
    yy = torch.mm(target, target.t())
    zz = torch.mm(gen_samples, target.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.1, 0.5, 1.0, 2.0]

        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [0.5, 1.0, 5.0, 10.0]

        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


def swd_vector(
    gen_samples: torch.Tensor,
    target: torch.Tensor,
    n_repeat_projection=128,
    proj_per_repeat=1000,
    device="cpu",
    seed=0
) -> float:
    """Sliced Wasserstein distance.

    Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2012). Wasserstein barycenter and its application
    to texture mixing. In Scale Space and Variational Methods in Computer Vision:
    Third International Conference, SSVM 2011, Ein-Gedi, Israel, May 29-June 2, 2011, Revised
    Selected Papers 3 (pp. 435-446). Springer Berlin Heidelberg.

    https://hal.science/hal-00476064/document
    """
    assert len(gen_samples.shape) == 2 and len(target.shape) == 2
    torch.manual_seed(seed)
    gen_samples, target = gen_samples.to(device), target.to(device)  # (B,d)

    distances = []
    for j in range(n_repeat_projection):
        # random unit vector.
        rand = torch.randn(gen_samples.size(1), proj_per_repeat).to(device)
        # rand = rand / torch.std(rand, dim=0, keepdim=True)
        rand = rand / torch.norm(rand, p=2, dim=0, keepdim=True)

        # projection.
        proj1 = torch.matmul(gen_samples, rand)  # (B,d) @ (d,proj_per_repeat) = (B,proj_per_repeat)
        proj2 = torch.matmul(target, rand)
        proj1, _ = torch.sort(proj1, dim=0)
        proj2, _ = torch.sort(proj2, dim=0)

        # d = torch.abs(proj1 - proj2)  # (B,proj_per_repeat)
        d = torch.norm(proj1-proj2, p=2, dim=0)
        distances.append(torch.mean(d))

    result = torch.mean(torch.stack(distances))
    return result




