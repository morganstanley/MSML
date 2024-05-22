from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime as dt
import torch

import colored_traceback.always

from runner import Runner
import util
import options


print(util.yellow("="*80))
print(util.yellow(f"\t\tTraining start at {dt.datetime.now().strftime('%m_%d_%Y_%H%M%S')}"))
print(util.yellow("="*80))
print(util.magenta("setting configurations..."))
opt = options.set()

def main(opt):
    run = Runner(opt)

    # ====== Training functions ======
    if opt.train_method == 'alternate':
        run.sb_alternate_train(opt)

    elif opt.train_method in ['alternate_imputation', 'alternate_imputation_v2']:
        run.sb_alternate_imputation_train(opt)

    elif opt.train_method == 'alternate_backward':
        run.sb_alternate_train_backward(opt)

    elif opt.train_method in ['alternate_backward_imputation', 'alternate_backward_imputation_v2']:
        run.sb_alternate_imputation_train_backward(opt)

    elif opt.train_method in ['dsm', 'dsm_v2', 'dsm_imputation', 'dsm_imputation_v2',
        'dsm_imputation_forward_verfication']:
        run.sb_dsm_train(opt)

    elif opt.train_method == 'joint':  # Deprecated.
        run.sb_joint_train(opt)

    elif opt.train_method == 'evaluation':  # skip training, directly go to full evaluation.
        util.compare_opts(opt, opt.ckpt_path)
        ckpt_file = os.path.join(opt.ckpt_path, opt.ckpt_file)
        util.restore_checkpoint(opt, run, ckpt_file)
        run.imputation_eval(opt, stage=None, quick_eval=False, run_validation=False,
            output_dir=opt.ckpt_path)

    else:
        raise NotImplementedError(f'New train method {opt.train_method}')

    # ====== Test functions ======
    # elif opt.compute_FID:
    #     run.evaluate(opt, util.get_load_it(opt.load), metrics=['FID','snapshot'])
    # elif opt.compute_NLL:
    #     run.compute_NLL(opt)
    # else:
    #     raise RuntimeError()

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
