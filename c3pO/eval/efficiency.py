import numpy as np


def _reg_interval_size(prediction, y, significance):
    idx = int(significance * 100 - 1)
    prediction = prediction[:, :, idx]

    return prediction[:, 1] - prediction[:, 0]


def reg_min_size(prediction, y, significance):
    return np.min(_reg_interval_size(prediction, y, significance))


def reg_q1_size(prediction, y, significance):
    return np.percentile(_reg_interval_size(prediction, y, significance), 25)


def reg_median_size(prediction, y, significance):
    return np.median(_reg_interval_size(prediction, y, significance))


def reg_q3_size(prediction, y, significance):
    return np.percentile(_reg_interval_size(prediction, y, significance), 75)


def reg_max_size(prediction, y, significance):
    return np.max(_reg_interval_size(prediction, y, significance))


def reg_mean_size(prediction, y, significance):
    """Calculates the average prediction interval size of a conformal
    regression model.
    """
    return np.mean(_reg_interval_size(prediction, y, significance))


def class_avg_c(prediction, y, significance):
    """Calculates the average number of classes per prediction of a conformal
    classification model.
    """
    prediction = prediction > significance
    return np.sum(prediction) / prediction.shape[0]


def class_mean_p_val(prediction, y, significance):
    """Calculates the mean of the p-values output by a conformal classification
    model.
    """
    return np.mean(prediction)


def class_one_c(prediction, y, significance):
    """Calculates the rate of singleton predictions (prediction sets containing
    only a single class label) of a conformal classification model.
    """
    prediction = prediction > significance
    n_singletons = np.sum(1 for _ in filter(lambda x: np.sum(x) == 1, prediction))
    return n_singletons / y.size


def class_empty(prediction, y, significance):
    """Calculates the rate of singleton predictions (prediction sets containing
    only a single class label) of a conformal classification model.
    """
    prediction = prediction > significance
    n_empty = np.sum(1 for _ in filter(lambda x: np.sum(x) == 0, prediction))
    return n_empty / y.size


def n_test(prediction, y, significance):
    """Provides the number of test patters used in the evaluation."""
    return y.size
