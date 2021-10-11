import abc
import numpy as np
import torch

EPS = 1e-6


class PLModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(self, x):
        return self.model(x)


class BaseScorer(abc.ABC):
    """Base scorer for Inductive Conformal Prediction"""

    @abc.abstractmethod
    def score(self, x, y=None):
        pass


class Normalizer(BaseScorer):
    def __init__(self, base_model, normalizer_model, err_func):
        super().__init__()
        self.base_model = base_model
        self.normalizer_model = normalizer_model
        self.err_func = err_func
        self.is_fitted = False

    def fit(self, x, y):
        residual_prediction = self.base_model.predict(x)
        residual_error = np.abs(self.err_func.apply(residual_prediction, y))
        residual_error += EPS  # Add small term to avoid log(0)
        log_err = np.log(residual_error)
        self.normalizer_model.fit(x, log_err)
        self.is_fitted = True

    def score(self, x, y=None):
        norm = np.exp(self.normalizer_model.predict(x))
        return norm
