import abc
import numpy as np


class ErrFunc(abc.ABC):
    """Base class for classification model error functions."""

    @abc.abstractmethod
    def apply(self, prediction, y):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
                Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
                True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
                Nonconformity scores of the samples.
        """
        pass


class BaseModelNC:
    """Base class for nonconformity scorers based on an underlying model.

    Parameters
    ----------
    model : Underlying pretrained model

    err_func : ClassificationErrFunc or RegressionErrFunc
            Error function object.

    normalizer : Normalizer
            Normalization model.

    beta : float
            Normalization smoothing parameter. As the beta-value increases,
            the normalized nonconformity function approaches a non-normalized
            equivalent.
    """

    def __init__(self, model, err_func, normalizer=None, beta=0):
        super().__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if self.normalizer is not None and hasattr(self.normalizer, "base_model"):
            self.normalizer.base_model = self.model

        self.clean = False

    def fit(self, x, y):
        """Fits the underlying model of the nonconformity scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
                Inputs of examples for fitting the underlying model.

        y : numpy array of shape [n_samples]
                Outputs of examples for fitting the underlying model.

        Returns
        -------
        None
        """
        if self.normalizer is not None and not self.normalizer.is_fitted:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
                Inputs of examples for which to calculate a nonconformity score.

        y : numpy array of shape [n_samples]
                Outputs of examples for which to calculate a nonconformity score.

        Returns
        -------
        nc : numpy array of shape [n_samples]
                Nonconformity scores of samples.
        """
        prediction = self.model.predict(x)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        return self.err_func.apply(prediction, y) / norm
