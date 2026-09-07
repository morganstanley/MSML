from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_metrics(log_path: Path):
    recs=[]
    for line in log_path.read_text().splitlines():
        if line.startswith('METRIC '):
            recs.append(json.loads(line.split(' ',1)[1]))
    return recs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('metrics_jsonl', type=str)
    ap.add_argument('--out', type=str, default=None)
    args=ap.parse_args()

    p=Path(args.metrics_jsonl)
    recs=load_metrics(p)
    if not recs:
        raise SystemExit('no records')

    t=[r.get('t',0) for r in recs]
    train_loss=[r.get('train_loss') for r in recs]

    fig,ax=plt.subplots(2,1,figsize=(8,7),sharex=True)
    ax[0].plot(t, train_loss, label='train_loss')
    ax[0].set_ylabel('loss (nats/token)')
    ax[0].grid(True, alpha=0.3)

    # plot val_bpb at eval points only
    te=[]
    ve=[]
    for r in recs:
        vb=r.get('val_bpb')
        if vb is not None:
            te.append(r.get('t',0))
            ve.append(vb)
    if ve:
        ax[1].plot(te, ve, marker='o', linewidth=1.5, label='val_bpb')
    ax[1].set_ylabel('val_bpb')
    ax[1].grid(True, alpha=0.3)

    ax[1].set_xlabel('seconds (training budget)')
    fig.suptitle(p.parent.name)

    out = Path(args.out) if args.out else (p.parent / 'curve.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


if __name__=='__main__':
    main()
