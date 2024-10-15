from lightning.pytorch.loggers import CSVLogger
from time_match.modules import (
    DDPMEstimator,
    BlendEstimator,
    FMEstimator,
    SGMEstimator,
    SIEstimator,
)

def get_estimator(args, dataset):
    logger = CSVLogger(save_dir=args.out_dir, name='logs')
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="gpu",
        devices=[args.device],
        num_sanity_val_steps=0,
    )

    if 'wiki' in args.dataset or 'taxi' in args.dataset:
        lags_seq = [1]
    else:
        lags_seq = None

    shared_kwargs = dict(
        freq=dataset.metadata.freq,
        prediction_length=args.prediction_length,
        input_size=args.max_target_dim,
        context_length=args.prediction_length,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        lr=args.lr,
        weight_decay=1e-8,
        dropout_rate=0.1,
        patience=10,
        scaling=args.scaling,
        lags_seq=lags_seq,
        num_parallel_samples=100,
        batch_size=args.batch_size,
        num_batches_per_epoch=50,
        denoising_model=args.denoising_model,
        velocity_model=args.velocity_model,
        rnn_model=args.rnn_model,
        trainer_kwargs=trainer_kwargs,
    )

    if args.estimator == 'ddpm':
        return DDPMEstimator(
            **shared_kwargs,
            linear_start=args.beta_start,
            linear_end=args.beta_end,
            n_timestep=args.steps,
        )
    elif args.estimator == 'blend':
        return BlendEstimator(
            **shared_kwargs,
            n_timestep=args.steps,
        )
    elif args.estimator == 'fm':
        return FMEstimator(
            **shared_kwargs,
            sigma_min=args.sigma_min,
            n_timestep=args.steps,
        )
    elif args.estimator == 'sgm':
        return SGMEstimator(
            **shared_kwargs,
            linear_start=args.beta_start,
            linear_end=args.beta_end,
            n_timestep=args.steps,
        )
    elif args.estimator == 'si':
        return SIEstimator(
            **shared_kwargs,
            epsilon=args.epsilon,
            start_noise=args.start_noise,
            interpolant=args.interpolant,
            gamma=args.gamma,
            n_timestep=args.steps,
            importance_sampling=args.importance_sampling,
        )
    else:
        raise NotImplementedError
