#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from mne import Epochs


# Define frequency bands
bands = {'delta': (0.5, 4),
         'theta': (4, 8),
         'alpha': (8, 13),
         'beta': (13, 30)}

def extract_psd_features(epoch):
    freqs = epoch.compute_psd().freqs
    band_inds = dict()
    for band, (fmin, fmax) in bands.items():
        band_inds[band] = np.where((freqs >= fmin) & (freqs <= fmax))

    psd_data = epoch.compute_psd().get_data()[0]
    band_power = dict()
    for band, inds in band_inds.items():
        band_power[band] = np.mean(psd_data[:, inds], axis=1)

    flat_power = np.concatenate([band_power[band].flatten() for band in bands], axis=0)
    return flat_power

def extract_time_domain_features(epoch):
    features = []
    data = epoch.get_data()
    for i in range(data.shape[1]):
        channel_data = data[:, i].squeeze()
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
        features.append(skew(channel_data))
        features.append(kurtosis(channel_data))
    #print(len(features))
    return features

def extract_wavelet_features(epoch):
    coeffs = []
    data = epoch.get_data()
    for i in range(data.shape[1]):
        channel_data = data[:, i].squeeze()
        cA, cD = pywt.dwt(channel_data, 'db4')  # Decompose signal using Daubechies 4 wavelet
        coeffs.append(cA.mean())
        coeffs.append(cA.std())
        coeffs.append(np.abs(cD).mean())  # Take absolute value of cD coefficients to account for negative values
        coeffs.append(np.abs(cD).std())
        coeffs.append(skew(np.abs(cD).flatten()))  # Skewness of cD coefficients
        coeffs.append(kurtosis(np.abs(cD).flatten()))  # Kurtosis of cD coefficients
        coeffs.append(np.sum(np.square(cA)))  # Energy of wavelet coefficients
        coeffs.append(np.sum(np.square(cD)))  # Energy of detail coefficients
        coeffs.append(-np.sum(np.square(cA) * np.log(np.square(cA))))  # Wavelet entropy of approximation coefficients
        coeffs.append(-np.sum(np.square(cD) * np.log(np.square(cD))))  # Wavelet entropy of detail coefficients
    #print(len(coeffs))
    return coeffs


def extract_combined_features(all_data):
    # Initialize an empty DataFrame to store the combined feature data
    combined_feature_df = pd.DataFrame()

    # Loop over each all_data object
    for i, data in enumerate(all_data):
        # Extract features
        psd_features = extract_psd_features(all_data[i])
        time_domain_features = extract_time_domain_features(all_data[i])
        wavelet_features = extract_wavelet_features(all_data[i])

        # Combine all features into a single list
        combined_features = np.hstack([psd_features, time_domain_features, wavelet_features])

        # Append the combined features as a row in the combined_feature_df dataframe
        combined_feature_df = combined_feature_df.append(pd.Series(combined_features), ignore_index=True)

    return combined_feature_df

