from dataclasses import dataclass, field
import pandas as pd
from .base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA

@dataclass
class ARIMAModel(BaseModel):
    order: tuple
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def train(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, start, end):
        return self.model_fit.predict(start=start, end=end)