import joblib
from dataclasses import dataclass

@dataclass
class BaseModel:
    model: any = None

    def train(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)