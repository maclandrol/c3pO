import abc
import numpy as np
import torch

EPS = 1e-6


class ModelWrapper:
    def __init__(self, model):
        self.last_x, self.last_y = None, None
        self.clean = False
        self.model = model

    def eval(self):
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def predict(self, x, *args, **kwargs):
        """Returns the prediction made by the underlying model.
        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of test examples.
        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted outputs of test examples.
        """
        self.eval()
        self.last_x = x
        self.last_y = self._predict(x, *args, **kwargs)
        return self.last_y.copy()

    @torch.no_grad()
    def _predict(self, x, *args, **kwargs):
        if isinstance(self.model, torch.nn.Module):
            return self.model(x)
        # for classifier takes predict_proba first
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        elif hasattr(self.model, "predict"):
            return self.model.predict(x)
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
