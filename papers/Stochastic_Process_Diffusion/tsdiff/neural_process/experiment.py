import seml
from sacred import Experiment
from tsdiff.neural_process.train import train

ex = Experiment()
seml.setup_logger(ex)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(seed: int, use_gp: bool, param: float):
    _, result = train(seed=seed, gp=use_gp, param=param)
    return result
