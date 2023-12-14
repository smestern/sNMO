from ..utils import *
from brian2 import *



class zErrorMetric():
    def __init__(self, y, shift=0):
        self.mean = 0
        self.std = 0
        self.shift = shift
        self.y = y
    def fit(self, ar, axis=0):
        self.mean = np.nanmean(ar, axis=axis)
        self.std = np.std(ar, axis=axis)
        self.y_scaled = self._zscore(self.y)
        #scaled_ar
        #error =  np.apply_along_axis(compute_ae, 1, ar,  self.y)
    def _zscore(self, ar): 
        zscored = (ar - self.mean)/ self.std
        return self.shift + zscored
    def transform(self, ar):
        error =  np.apply_along_axis(compute_se, 1, self._zscore(ar),  self.y_scaled)
        return np.nanmean(error, axis=1)
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

##TODO
class thresholdErrorMetric():
    def __init__(self, y=None, thresholds=[None, None]):
        raise NotImplementedError
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