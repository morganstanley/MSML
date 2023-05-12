from main_model import CSDI_Physio
from dataset_physio import get_dataloader
from utils import train, evaluate

import seml
from sacred import Experiment

ex = Experiment()
seml.setup_logger(ex)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    nsample: int,
    testmissingratio: float,
    gp_noise: bool,
    is_unconditional: bool,
    timeemb: int,
    featureemb: int,
    target_strategy: str,
    num_steps: int,
    schedule: str,
    beta_start: float,
    beta_end: float,
    layers: int,
    channels: int,
    nheads: int,
    diffusion_embedding_dim: int,
    gp_sigma: float = None,
    device: str = 'cuda:0',
):

    config = dict(
        train=dict(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
        ),
        model=dict(
            timeemb=timeemb,
            featureemb=featureemb,
            is_unconditional=is_unconditional,
            target_strategy=target_strategy,
            test_missing_ratio=testmissingratio,
        ),
        diffusion=dict(
            num_steps=num_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            layers=layers,
            channels=channels,
            nheads=nheads,
            diffusion_embedding_dim=diffusion_embedding_dim,
        ),
    )

    train_loader, valid_loader, test_loader = get_dataloader(
        seed=seed,
        nfold=0,
        batch_size=batch_size,
        missing_ratio=testmissingratio,
    )

    model = CSDI_Physio(config, device, gp_noise=gp_noise, gp_sigma=gp_sigma).to(device)

    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername='',
    )

    results = evaluate(model, test_loader, nsample=nsample, scaler=1, foldername='')
    return results
