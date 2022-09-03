import os

import catboost


class CatBoostClf:
    def __init__(self, model_path: str):
        self._model = catboost.CatBoostClassifier()
        self._model.load_model(model_path)
        self.DEFAULT_LABEL = 0

    def predict(self, features):
        if type(features) == int and features == -1:
            return self.DEFAULT_LABEL

        return self._model.predict(features.values)
