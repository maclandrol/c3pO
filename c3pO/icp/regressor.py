import numpy as np
from c3pO.icp._base import BaseICP
from c3pO.base import BaseScorer


class ICPRegressor(BaseICP):
    """Inductive conformal regressor.

    Parameters
    ----------
    nc_function : BaseScorer
            Nonconformity scorer object used to calculate nonconformity of
            calibration examples and test patterns. Should implement ``fit(x, y)``,
            ``calc_nc(x, y)`` and ``predict(x, nc_scores, significance)``.

    Attributes
    ----------
    cal_x : numpy array of shape [n_cal_examples, n_features]
            Inputs of calibration set.

    cal_y : numpy array of shape [n_cal_examples]
            Outputs of calibration set.

    nc_function : BaseScorer
            Nonconformity scorer object used to calculate nonconformity scores.

    See also
    --------
    IcpClassifier

    References
    ----------
    .. [1] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).
            Inductive confidence machines for regression. In Machine Learning: ECML
            2002 (pp. 345-356). Springer Berlin Heidelberg.

    .. [2] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
            intervals with regression neural networks. Neural Networks, 24(8),
            842-851.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from nonconformist.base import RegressorAdapter
    >>> from nonconformist.icp import ICPRegressor
    >>> from nonconformist.nc import RegressorNc, AbsErrorErrFunc
    >>> boston = load_boston()
    >>> idx = np.random.permutation(boston.target.size)
    >>> train = idx[:int(idx.size / 3)]
    >>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
    >>> test = idx[int(2 * idx.size / 3):]
    >>> model = RegressorAdapter(DecisionTreeRegressor())
    >>> nc = RegressorNc(model, AbsErrorErrFunc())
    >>> icp = ICPRegressor(nc)
    >>> icp.fit(boston.data[train, :], boston.target[train])
    >>> icp.calibrate(boston.data[cal, :], boston.target[cal])
    >>> icp.predict(boston.data[test, :], significance=0.10)
    ...     # doctest: +SKIP
    array([[  5. ,  20.6],
            [ 15.5,  31.1],
            ...,
            [ 14.2,  29.8],
            [ 11.6,  27.2]])
    """

    def __init__(self, nc_function, condition=None):
        super().__init__(nc_function, condition)

    def predict(self, x, significance=None):
        """Predict the output values for a set of input patterns.

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
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99}
                If significance is ``None``, then p contains the interval (minimum
                and maximum boundaries) for each test pattern, and each significance
                level (0.01, 0.02, ..., 0.99). If significance is a float between
                0 and 1, then p contains the prediction intervals (minimum and
                maximum	boundaries) for the set of test patterns at the chosen
                significance level.
        """
        # TODO: interpolated p-values

        n_significance = 99 if significance is None else np.array(significance).size

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None)) for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], self.cal_scores[condition], significance)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction
