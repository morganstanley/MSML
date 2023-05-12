import seml
from sacred import Experiment
from tsdiff.synthetic.train import train

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
    model: str,
    diffusion: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    predict_gaussian_noise: bool,
    gp_sigma: float = None,
    ou_theta: float = None,
):
    results = train(**locals())
    return results
