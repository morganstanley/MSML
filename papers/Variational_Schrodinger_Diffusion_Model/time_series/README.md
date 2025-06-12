## Installation

requirement: 
```bash
gluonts==0.14.3
git+https://github.com/zalandoresearch/pytorch-ts.git@version-0.7.0
pytorchts or pytorchts==0.6.0
```

Train VSDM

```
python main.py --data solar_nips --seed 1 --batch_size 64 --epochs 20 --forward_opt_steps 200 --backward_opt_steps 200
```

Train DSM

```
python main.py --data electricity_nips --seed 1 --batch_size 32 --hidden_dim 64 --epochs 20 --forward_opt_steps 0 --backward_opt_steps 200 --t0 0.01 --T 1 --beta_min 0.1 --beta_max 10 --beta_r 1.7 --steps 100 --device 1
```

Train DDPM

```
python main.py --data constant --seed 1 --batch_size 32 --hidden_dim 64 --epochs 20 --beta_min 1e-4 --beta_max 0.1 --steps 150 --ddpm
```

Credit to: [Marin Bilos](https://github.com/mbilos)
