#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew
from mne import Epochs
from joblib import Parallel, delayed
from tqdm import tqdm


# ... (all your functions and definitions remain unchanged) ...

# Define frequency bands
bands = {'delta': (0.5, 4),
         'theta': (4, 8),
         'alpha': (8, 13),
         'beta': (13, 30)}


def extract_psd_features(epoch):
    # Get the frequency axis from the PSD data
    freqs = epoch.compute_psd().freqs

    # Find the frequency bin indices corresponding to each band
    band_inds = dict()
    for band, (fmin, fmax) in bands.items():
        band_inds[band] = np.where((freqs >= fmin) & (freqs <= fmax))

    # Compute the power in each frequency band
    psd_data = epoch.compute_psd().get_data()[0]  # Assuming one epoch only
    band_power = dict()
    for band, inds in band_inds.items():
        if len(inds[0]) > 0:
            band_power[band] = np.mean(psd_data[:, inds], axis=1).flatten()
        else:
            band_power[band] = np.zeros(psd_data.shape[0])  # Set power to 0 if no frequencies found in the band

    # Aggregate the power values across channels for each frequency band
    aggregated_power = {band: np.mean(band_power[band]) for band in bands}
    return aggregated_power


def extract_time_domain_features(epoch):
    features = []
    data = epoch.get_data()
    mean_data = np.mean(data, axis=1)
    std_data = np.std(data, axis=1)
    skew_data = skew(data, axis=1)
    kurtosis_data = kurtosis(data, axis=1)
    features.append(np.mean(mean_data))
    features.append(np.mean(std_data))
    features.append(np.mean(skew_data))
    features.append(np.mean(kurtosis_data))
    return features


def extract_wavelet_features(epoch):
    coeffs = []
    data = epoch.get_data()
    for i in range(data.shape[1]):
        channel_data = data[:, i].squeeze()
        cA, cD = pywt.dwt(channel_data, 'db4')  # Decompose signal using Daubechies 4 wavelet
        coeffs.append({
            'cA_mean': cA.mean(),
            'cA_std': cA.std(),
            'cD_abs_mean': np.abs(cD).mean(),
            'cD_abs_std': np.abs(cD).std(),
            'cD_abs_skew': skew(np.abs(cD).flatten()),
            'cD_abs_kurtosis': kurtosis(np.abs(cD).flatten()),
            'cA_energy': np.sum(np.square(cA)),
            'cD_energy': np.sum(np.square(cD)),
            'cA_entropy': -np.sum(np.square(cA) * np.log(np.square(cA))),
            'cD_entropy': -np.sum(np.square(cD) * np.log(np.square(cD))),
        })

    mean_coeffs = {key: np.mean([coef[key] for coef in coeffs]) for key in coeffs[0]}
    return mean_coeffs


def process_epoch(i, epoch):
    # Extract features
    psd_features = extract_psd_features(epoch)
    time_domain_features = extract_time_domain_features(epoch)
    wavelet_features = extract_wavelet_features(epoch)

    # Combine all features into a single dictionary
    combined_features = {**psd_features, 'mean': time_domain_features[0], 'std': time_domain_features[1],
                         'skew': time_domain_features[2], 'kurtosis': time_domain_features[3], **wavelet_features}

    return i, combined_features


def extract_features(all_data):
    # Initialize an empty DataFrame to store the feature data
    feature_df = pd.DataFrame()

    # Use joblib to parallelize feature extraction
    results = Parallel(n_jobs=-1)(delayed(process_epoch)(i, all_data[i]) for i, epoch in enumerate(all_data))
    # Use joblib to parallelize feature extraction
    results = Parallel(n_jobs=-1)(delayed(process_epoch)(i, all_data[i]) for i in tqdm(range(len(all_data)), desc="Extracting features"))

    # Collect the results and add them to the feature_df dataframe
    for i, combined_features in results:
        # Convert combined_features dictionary to a DataFrame
        combined_features_df = pd.DataFrame(combined_features, index=[i])

        # Append the combined_features_df dataframe as a row in the feature_df dataframe
        feature_df = feature_df.append(combined_features_df, ignore_index=True)

    return feature_df

