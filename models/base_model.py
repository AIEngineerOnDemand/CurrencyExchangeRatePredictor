class BaseModel:
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclass must implement this method")

    def predict(self, X_test):
        raise NotImplementedError("Subclass must implement this method")