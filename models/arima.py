from .base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel(BaseModel):
    def __init__(self, order):
        self.order = order
        self.model = None

    def train(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, start, end):
        return self.model_fit.predict(start=start, end=end)