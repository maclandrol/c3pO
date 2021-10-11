from collections import defaultdict
from functools import partial
import numpy as np


class BaseICP:
    """Base class for inductive conformal predictors."""

    def __init__(self, nc_function, condition=None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        default_condition = lambda x: 0
        is_default = callable(condition) and (
            condition.__code__.co_code == default_condition.__code__.co_code
        )

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

    def calibrate(self, x, y, increment=False):
        """Calibrate conformal predictor based on underlying nonconformity
        scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
                Inputs of examples for calibrating the conformal predictor.

        y : numpy array of shape [n_samples, n_features]
                Outputs of examples for calibrating the conformal predictor.

        increment : boolean
                If ``True``, performs an incremental recalibration of the conformal
                predictor. The supplied ``x`` and ``y`` are added to the set of
                previously existing calibration examples, and the conformal
                predictor is then calibrated on both the old and new calibration
                examples.

        Returns
        -------
        None
        """
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i])) for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :], self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
            self.cal_scores = {0: np.sort(cal_scores)[::-1]}

    def _calibrate_hook(self, x, y, increment):
        pass

    def _update_calibration_set(self, x, y, increment):
        if increment and self.cal_x is not None and self.cal_y is not None:
            self.cal_x = np.vstack([self.cal_x, x])
            self.cal_y = np.hstack([self.cal_y, y])
        else:
            self.cal_x, self.cal_y = x, y
