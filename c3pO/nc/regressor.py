import abc
import numpy as np
from c3pO.nc._base import ErrFunc, BaseModelNC


class RegressionErrFunc(ErrFunc):
    """Base class for regression model error functions."""

    @abc.abstractmethod
    def apply_inverse(self, nc, significance, *args, **kwargs):
        """Apply the inverse of the nonconformity function (i.e.,
        calculate prediction interval).

        Parameters
        ----------
        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (0, 1).

        Returns
        -------
        interval : numpy array of shape [n_samples, 2]
            Minimum and maximum interval boundaries for each prediction.
        """
        pass


class AbsErrorErrFunc(RegressionErrFunc):
    """Calculates absolute error nonconformity for regression problems.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        | y_i - \hat{y}_i |
    """

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        # TODO: should probably warn against too few calibration examples
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):
    """Calculates signed error nonconformity for regression problems.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        y_i - \hat{y}_i

    References
    ----------
    .. [1] Linusson, Henrik, Ulf Johansson, and Tuve Lofstrom.
        Signed-error conformal regression. Pacific-Asia Conference on Knowledge
        Discovery and Data Mining. Springer International Publishing, 2014.
    """

    def apply(self, prediction, y):
        return prediction - y

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        upper = int(np.floor((significance / 2) * (nc.size + 1)))
        lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
        # TODO: should probably warn against too few calibration examples
        upper = min(max(upper, 0), nc.size - 1)
        lower = max(min(lower, nc.size - 1), 0)
        return np.vstack([-nc[lower], nc[upper]])


class RegressorNC(BaseModelNC):
    """Nonconformity scorer using an underlying regression model.

    Parameters
    ----------
    model : Model
        Underlying regression model used for calculating nonconformity scores.

    err_func : RegressionErrFunc
        Error function object.

    normalizer : Normalizer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : Model
        Underlying model object.

    err_func : RegressionErrFunc
        Scorer function used to calculate nonconformity scores.
    """

    def __init__(self, model, err_func=AbsErrorErrFunc(), normalizer=None, beta=0):

        if not isinstance(err_func, RegressionErrFunc):
            raise ValueError("Only classification error function supported")
        super().__init__(model, err_func, normalizer, beta)

    def predict(self, x, nc, significance=None):
        """Constructs prediction intervals for a set of test examples.

        Predicts the output of each test pattern using the underlying model,
        and applies the (partial) inverse nonconformity function to each
        prediction, resulting in a prediction interval for each test pattern.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        significance : float
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then intervals for
            all significance levels (0.01, 0.02, ..., 0.99) are output in a
            3d-matrix.

        Returns
        -------
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each significance
            level (0.01, 0.02, ..., 0.99). If significance is a float between
            0 and 1, then p contains the prediction intervals (minimum and
            maximum	boundaries) for the set of test patterns at the chosen
            significance level.
        """
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            err_dist *= norm

            intervals[:, 0] = prediction - err_dist[0, :]
            intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals
