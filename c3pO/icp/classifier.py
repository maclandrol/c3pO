import numpy as np
from c3pO.icp._base import BaseICP
from c3pO.base import BaseScorer
from c3pO.utils import calc_p


class ICPClassifier(BaseICP):
    """Inductive conformal classifier.

    Parameters
    ----------
    nc_function : BaseScorer
            Nonconformity scorer object used to calculate nonconformity of
            calibration examples and test patterns. Should implement ``fit(x, y)``
            and ``calc_nc(x, y)``.

    smoothing : boolean
            Decides whether to use stochastic smoothing of p-values.

    Attributes
    ----------
    cal_x : numpy array of shape [n_cal_examples, n_features]
            Inputs of calibration set.

    cal_y : numpy array of shape [n_cal_examples]
            Outputs of calibration set.

    nc_function : BaseScorer
            Nonconformity scorer object used to calculate nonconformity scores.

    classes : numpy array of shape [n_classes]
            List of class labels, with indices corresponding to output columns
             of ICPClassifier.predict()

    See also
    --------
    IcpRegressor

    References
    ----------
    .. [1] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
            intervals with regression neural networks. Neural Networks, 24(8),
            842-851.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from nonconformist.base import ClassifierAdapter
    >>> from nonconformist.icp import ICPClassifier
    >>> from nonconformist.nc import ClassifierNc, MarginErrFunc
    >>> iris = load_iris()
    >>> idx = np.random.permutation(iris.target.size)
    >>> train = idx[:int(idx.size / 3)]
    >>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
    >>> test = idx[int(2 * idx.size / 3):]
    >>> model = ClassifierAdapter(DecisionTreeClassifier())
    >>> nc = ClassifierNc(model, MarginErrFunc())
    >>> icp = ICPClassifier(nc)
    >>> icp.fit(iris.data[train, :], iris.target[train])
    >>> icp.calibrate(iris.data[cal, :], iris.target[cal])
    >>> icp.predict(iris.data[test, :], significance=0.10)
    ...             # doctest: +SKIP
    array([[ True, False, False],
            [False,  True, False],
            ...,
            [False,  True, False],
            [False,  True, False]], dtype=bool)
    """

    def __init__(self, nc_function, condition=None, smoothing=True):
        super().__init__(nc_function, condition)
        self.classes = None
        self.smoothing = smoothing

    def _calibrate_hook(self, x, y, increment=False):
        self._update_classes(y, increment)

    def _update_classes(self, y, increment):
        if self.classes is None or not increment:
            self.classes = np.unique(y)
        else:
            self.classes = np.unique(np.hstack([self.classes, y]))

    def predict(self, x, significance=None):
        """Predict the output values for a set of input patterns.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
                Inputs of patters for which to predict output values.

        significance : float or None
                Significance level (maximum allowed error rate) of predictions.
                Should be a float between 0 and 1. If ``None``, then the p-values
                are output rather than the predictions.

        Returns
        -------
        p : numpy array of shape [n_samples, n_classes]
                If significance is ``None``, then p contains the p-values for each
                sample-class pair; if significance is a float between 0 and 1, then
                p is a boolean array denoting which labels are included in the
                prediction sets.
        """
        # TODO: if x == self.last_x ...
        n_test_objects = x.shape[0]
        p = np.zeros((n_test_objects, self.classes.size))

        ncal_ngt_neq = self._get_stats(x)

        for i in range(len(self.classes)):
            for j in range(n_test_objects):
                p[j, i] = calc_p(
                    ncal_ngt_neq[j, i, 0],
                    ncal_ngt_neq[j, i, 1],
                    ncal_ngt_neq[j, i, 2],
                    self.smoothing,
                )

        if significance is not None:
            return p > significance
        else:
            return p

    def _get_stats(self, x):
        n_test_objects = x.shape[0]
        ncal_ngt_neq = np.zeros((n_test_objects, self.classes.size, 3))
        for i, c in enumerate(self.classes):
            test_class = np.zeros(x.shape[0], dtype=self.classes.dtype)
            test_class.fill(c)

            # TODO: maybe calculate p-values using cython or similar
            # TODO: interpolated p-values

            # TODO: nc_function.calc_nc should take X * {y1, y2, ... ,yn}
            test_nc_scores = self.nc_function.score(x, test_class)
            for j, nc in enumerate(test_nc_scores):
                cal_scores = self.cal_scores[self.condition((x[j, :], c))][::-1]
                n_cal = cal_scores.size

                idx_left = np.searchsorted(cal_scores, nc, "left")
                idx_right = np.searchsorted(cal_scores, nc, "right")

                ncal_ngt_neq[j, i, 0] = n_cal
                ncal_ngt_neq[j, i, 1] = n_cal - idx_right
                ncal_ngt_neq[j, i, 2] = idx_right - idx_left

        return ncal_ngt_neq

    def predict_conf(self, x):
        """Predict the output values for a set of input patterns, using
        the confidence-and-credibility output scheme.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
                Inputs of patters for which to predict output values.

        Returns
        -------
        p : numpy array of shape [n_samples, 3]
                p contains three columns: the first column contains the most
                likely class for each test pattern; the second column contains
                the confidence in the predicted class label, and the third column
                contains the credibility of the prediction.
        """
        p = self.predict(x, significance=None)
        label = p.argmax(axis=1)
        credibility = p.max(axis=1)
        for i, idx in enumerate(label):
            p[i, idx] = -np.inf
        confidence = 1 - p.max(axis=1)

        return np.array([label, confidence, credibility]).T
