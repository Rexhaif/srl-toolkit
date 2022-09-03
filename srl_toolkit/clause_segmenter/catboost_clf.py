import os

import catboost


class CatBoostClf:
    def __init__(self, model_dir_path):
        self._model = catboost.CatBoostClassifier()
        self._model.load_model(os.path.join(model_dir_path, "catboost_clf_small.pkl"))
        self.DEFAULT_LABEL = 0

    def predict(self, features):
        if type(features) == int and features == -1:
            return self.DEFAULT_LABEL

        return self._model.predict(features.values)
