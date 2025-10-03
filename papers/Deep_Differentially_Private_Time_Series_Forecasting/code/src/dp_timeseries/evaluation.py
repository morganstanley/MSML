from gluonts.dataset.common import Dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Predictor
from gluonts.torch import PyTorchLightningEstimator
from gluonts.transform._base import Transformation
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


def eval_predictor(dataset: Dataset,
                   predictor: Predictor) -> dict:
    """Computes gluonts evaluation metrics on dataset.

    Args:
        dataset (Dataset): Test / Validation dataset to evaluate on.
        predictor (Predictor): Predictor to use.

    Returns:
        dict: Dict of metric name -- metric value pairs.
    """

    # Do forecasting
    forecast_it, target_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor)

    forecasts = list(forecast_it)
    targets = list(target_it)

    # TODO: Get iTransformer to work for both univariate and multivariate
    for forecast in forecasts:
        if isinstance(forecast, SampleForecast) and (forecast.samples.ndim == 3):
            forecast.samples = forecast.samples.squeeze(-1)

    # Compute metrics
    evaluator = Evaluator(num_workers=0)
    metrics, _ = evaluator(targets, forecasts)

    return metrics


class EvaluateCallback(Callback):
    """Callback for computing metrics via predictor corresponding to module.
    """
    def __init__(self,
                 dataset: Dataset,
                 transformation: Transformation,
                 estimator: PyTorchLightningEstimator,
                 eval_every_n_epoch: None | int = None) -> None:
        """
        Args:
            dataset (Dataset): Dataset generated with set_train=False.
            transformation (Transformation): Transformation to be used by predictor.
            estimator (PyTorchLightningEstimator): Estimator that
                wraps LightningModule and PytorchPredictor.
            eval_every_n_epoch (None | int, optional): After how many epochs
                this callback should run.
                If None, will be run whenever Trainer performs validation.
                Defaults to None.
        """

        super().__init__() 
        self.dataset = dataset
        self.transformation = transformation
        self.estimator = estimator
        self.eval_every_n_epoch = eval_every_n_epoch

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.eval_every_n_epoch is not None:
            return

        self.eval(pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.eval_every_n_epoch is None:
            return

        if (pl_module.current_epoch + 1) % self.eval_every_n_epoch == 0:

            pl_module.eval()
            assert pl_module.training is False

            self.eval(pl_module)

            pl_module.train()

    def eval(self, pl_module: LightningModule) -> None:
        predictor = self.estimator.create_predictor(
            self.transformation, pl_module)

        metrics = eval_predictor(self.dataset, predictor)

        pl_module.log_dict({
            f'val_{k}': v for k, v in metrics.items()
        })
