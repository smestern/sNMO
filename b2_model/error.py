from utils import *
from brian2 import *
from b2_model.brian2_model import brian2_model
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects


class zErrorMetric():
    def __init__(self, y, shift=100):
        self.mean = 0
        self.std = 0
        self.shift = shift
        self.y = y
    def fit(self, ar, axis=0):
        error =  np.apply_along_axis(compute_se, 1, ar,  self.y)
        self.mean = np.nanmean(error, axis=0)
        self.std = np.std(error, axis=0)
    def _zscore(self, ar): 
        zscored = (ar - self.mean)/ self.std
        return self.shift + zscored
    def transform(self, ar):
        error =  np.apply_along_axis(compute_se, 1, ar,  self.y)
        zerror = self._zscore(error)
        return np.nanmean(zerror, axis=1)
    def fit_transform(self, ar):
        self.fit(ar)
        error = self.transform(ar)
        return error

class weightedErrorMetric():
    def __init__(self, y=None, weights=None, splits=None, infer_weights=False):
        self.y = y
        self.splits = splits
        self.infer_weights = infer_weights
        if weights is None:
            self._weights = np.full(y.shape[0], 1)
            self.weights = self._weights
        else:
            self._weights = weights
            self.weights = weights

        if self.splits is not None:
            self.weights = self._gen_weights_from_splits()

    def fit(self, ar, axis=0):
        if self.infer_weights:
            #compute col mean
            error =  np.sqrt(np.apply_along_axis(compute_se, 1, ar,  self.y))
            col_mean = np.nanmean(error, axis=0)
            #just scale everything to the last column?
            self.weights = col_mean / col_mean[-1]
        else:
            pass
    def _gen_weights_from_splits(self):
        out_weights = []
        for split, w in zip(self.splits, self._weights):
            len_split = np.int(split[1] - split[0])
            out_weights.append(np.full(len_split, w))
        return np.hstack(out_weights)
    def _weighted_sum(self, ar): 
        e_sum = []
        for col, w in zip(ar.T, self.weights):
            e_sum.append(col*w)
        e_sum = np.vstack(e_sum).T
        return np.sum(e_sum, axis=1)
    def transform(self, ar):
        if self.y is not None:
            error =  np.apply_along_axis(compute_se, 1, ar,  self.y)
        else:
            error = ar
        werror = self._weighted_sum(error)
        return werror
    def fit_transform(self, ar):
        self.fit(ar)
        error = self.transform(ar)
        return error


class thresholdErrorMetric():
    def __init__(self, y=None, thresholds=[None, None]):
        self.y = y
    def fit(self):
        pass
    def transform(self, ar):
        #compute a poor mans mask?
        for col, thres in zip(ar.T, thresholds):
            if thres is not None:
                col[col>=thres] = 999
        return np.sum(ar, axis=1)

    def fit_transform(self, ):
        pass