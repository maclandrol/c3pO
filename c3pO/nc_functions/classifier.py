import numpy as np
from c3pO.nc_functions._base import ErrFunc, BaseModelNc


class ClassificationErrFunc(ErrFunc):
    """Base class for classification model error functions."""

    pass


class InverseProbabilityErrFunc(ClassificationErrFunc):
    """Calculates the probability of not predicting the correct class.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        1 - \hat{P}(y_i | x) \, .
    """

    def apply(self, probs, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= probs.shape[1]:
                prob[i] = 0
            else:
                prob[i] = probs[i, int(y_)]
        return 1 - prob


class MarginErrFunc(ClassificationErrFunc):
    """
    Calculates the margin error.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        0.5 - \dfrac{\hat{P}(y_i | x) - max_{y \, != \, y_i} \hat{P}(y | x)}{2}
    """

    def apply(self, probs, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= probs.shape[1]:
                prob[i] = 0
            else:
                prob[i] = probs[i, int(y_)]
                probs[i, int(y_)] = -np.inf
        return 0.5 - ((prob - probs.max(axis=1)) / 2)


class ClassifierNc(BaseModelNc):
    """Nonconformity scorer using an underlying class probability estimating
    model.

    Parameters
    ----------
    model : ClassifierAdapter
            Underlying classification model used for calculating nonconformity
            scores.

    err_func : ClassificationErrFunc
            Error function object.

    normalizer : BaseScorer
            Normalization model.

    beta : float
            Normalization smoothing parameter. As the beta-value increases,
            the normalized nonconformity function approaches a non-normalized
            equivalent.

    Attributes
    ----------
    model : ClassifierAdapter
            Underlying model object.

    err_func : ClassificationErrFunc
            Scorer function used to calculate nonconformity scores.

    See also
    --------
    RegressorNc, NormalizedRegressorNc
    """

    def __init__(self, model, err_func=MarginErrFunc(), normalizer=None, beta=0):
        if not isinstance(err_func, ClassificationErrFunc):
            raise ValueError("Only classification error function supported")
        super().__init__(model, err_func, normalizer, beta)
