from .base_model import BaseModel
from statsmodels.tsa.arima_model import ARIMA

class ARIMAModel(BaseModel):
    def __init__(self, order):
        self.model = ARIMA(order=order)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)