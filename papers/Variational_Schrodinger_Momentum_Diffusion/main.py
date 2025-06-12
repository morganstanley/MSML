from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from runner import Runner
import util
import options

import colored_traceback.always
from ipdb import set_trace as debug

print(util.yellow("======================================================="))
print(util.yellow("     Likelihood-Training of Schrodinger Bridge"))
print(util.yellow("======================================================="))
print(util.magenta("setting configurations..."))
opt = options.set()

def main(opt):
    run = Runner(opt)

    # ====== Training functions ======
    run.sb_alternate_train(opt)

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
