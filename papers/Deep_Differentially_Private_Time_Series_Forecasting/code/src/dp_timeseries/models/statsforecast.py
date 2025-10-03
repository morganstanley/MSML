from copy import deepcopy
from typing import List, Optional
from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import QuantileForecast
from gluonts.ext.statsforecast import AutoARIMAPredictor, AutoETSPredictor
from gluonts.transform._base import MapTransformation


class PreprocessedAutoARIMAPredictor(AutoARIMAPredictor):
    def __init__(
        self,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        input_transform: Optional[MapTransformation] = None,
        **kwargs
    ) -> None:
        super().__init__(prediction_length, quantile_levels, **kwargs)
        self.input_transform = input_transform

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        if self.input_transform is not None:
            item = deepcopy(entry)
            item = self.input_transform.map_transform(item, is_train=False)

        return super().predict_item(item)


class PreprocessedAutoETSPredictor(AutoETSPredictor):
    def __init__(
        self,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        input_transform: Optional[MapTransformation] = None,
        **kwargs
    ) -> None:
        super().__init__(prediction_length, quantile_levels, **kwargs)
        self.input_transform = input_transform

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        if self.input_transform is not None:
            item = deepcopy(entry)
            item = self.input_transform.map_transform(item, is_train=False)

        return super().predict_item(item)
