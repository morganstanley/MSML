clear

# plot kl-divergence
python run_flower.py --num_points 100 --M 5 --warm_up 1000 --plot_after 500 --n_epoch 100000

# multi-modal simulation
python sample_with_bound.py --optimizer 'sgld' --lr 5e-4 --num_points 1000 --save_after 10 --n_epoch 500000 --domain_type flower --if_include_domain True --radius 2
python sample_with_bound.py --optimizer 'cyclic_sgld' --lr 1e-3 --M 5 --num_points 1000 --save_after 10 --n_epoch 500000 --domain_type flower --if_include_domain True --radius 2
python sample_with_bound.py --optimizer 'resgld' --lr 5e-4 --lr_gap 3.0 --hat_var 9.5 --num_points 1000 --save_after 10 --n_epoch 500000 --domain_type flower --if_include_domain True --radius 2
