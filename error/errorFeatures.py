import numpy
from ipfx import feature_extractor

from utils.snm_utils import add_spikes_to_voltage


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
