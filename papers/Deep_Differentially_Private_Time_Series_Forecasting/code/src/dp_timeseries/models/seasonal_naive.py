from copy import deepcopy
from typing import Callable, Optional, Union

from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import Forecast
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.transform import LastValueImputation, MissingValueImputation
from gluonts.transform._base import MapTransformation


class PreprocessedSeasonalNaivePredictor(SeasonalNaivePredictor):
    def __init__(
        self,
        prediction_length: int,
        season_length: Union[int, Callable],
        imputation_method: MissingValueImputation = LastValueImputation(),
        input_transform: Optional[MapTransformation] = None
    ) -> None:
        super().__init__(prediction_length, season_length, imputation_method)
        self.input_transform = input_transform

    def predict_item(self, item: DataEntry) -> Forecast:
        if self.input_transform is not None:
            item = deepcopy(item)
            item = self.input_transform.map_transform(item, is_train=False)

        return super().predict_item(item)
