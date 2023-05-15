import seml
from sacred import Experiment
from tsdiff.forecasting.train import train

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
    dataset: str,
    network: str,
    noise: bool,
    diffusion_steps: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_cells: int,
    hidden_dim: int,
    residual_layers: int,
):
    results = train(**locals())
    return results
