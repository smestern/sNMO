import numpy as np
from ipfx import feature_extractor
from ..utils import compute_FI_curve, compute_rmp, compute_steady_hyp, compute_min_stim,add_spikes_to_voltage


def model_feature_curve(model):
    real_fi, real_isi = compute_FI_curve(model.spike_times, model._run_time) #Compute the real FI curve
    real_fi = np.hstack((real_fi, real_isi))
    real_rmp = compute_rmp(model.realY, model.realC)
    real_min = []
    real_subt = []
    for x in  model.subthresholdSweep :
        temp = compute_steady_hyp(model.realY[x, :].reshape(1,-1), model.realC[x, :].reshape(1,-1))
        temp_min = compute_min_stim(model.realY[x, :], model.realX[x,:], strt=0.62, end=1.2)
        real_subt.append(temp)
        real_min.append(temp_min)
    real_subt = np.array(real_subt)        
    
    real_min = np.hstack(real_min)
    
    np_o = np.hstack((real_fi, real_rmp, real_subt, real_min))
    
    return np_o


class FeatureExtractor():
    """Feature extractor for error analysis"""

    def __init__(self):
        return self

    def extract_features(self, t, v, i, s, dt, start, end, impute_spikes=False):
        """_summary_

        Args:
            t (_type_): _description_
            v (_type_): _description_
            i (_type_): _description_
            s (_type_): _description_
            dt (_type_): _description_
            start (_type_): _description_
            end (_type_): _description_
            impute_spikes (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """      
        features = {}

        if impute_spikes:
            s = add_spikes_to_voltage(t, v, s, dt)

        # extract spike features
        spike_features = self._extract_spike_features(t, v, i, s, dt, start, end)
        features.update(spike_features)

        # extract waveform features
        waveform_features = self._extract_waveform_features(t, v, i, s, dt, start, end)
        features.update(waveform_features)

        # extract subthresh features
        subthresh_features = self._extract_subthresh_features(t, v, i, s, dt, start, end)
        features.update(subthresh_features)

        return features
    
    def _extract_spike_features(self, t, v, i, s, dt, start, end):
        pass