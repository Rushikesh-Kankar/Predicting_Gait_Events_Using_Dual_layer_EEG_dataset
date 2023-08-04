#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#trailcell

# import numpy as np
# import pandas as pd
# import pywt


# from scipy.stats import skew, kurtosis
# from mne import Epochs
# from scipy.signal import correlate
# from joblib import Parallel, delayed
# from scipy.stats import entropy
# from pyentrp import entropy as ent
# from tqdm import tqdm

# # Define frequency bands
# bands = {'delta': (0.5, 4),
#          'theta': (4, 8),
#          'alpha': (8, 13),
#          'beta': (13, 30)}

# def extract_psd_features(epoch):
#     freqs = epoch.compute_psd().freqs
#     band_inds = dict()
#     for band, (fmin, fmax) in bands.items():
#         band_inds[band] = np.where((freqs >= fmin) & (freqs <= fmax))

#     psd_data = epoch.compute_psd().get_data()[0]
#     band_power = dict()
#     for band, inds in band_inds.items():
#         band_power[band] = np.mean(psd_data[:, inds], axis=1)

#     flat_power = np.concatenate([band_power[band].flatten() for band in bands], axis=0)
#     return flat_power

# def extract_time_domain_features(epoch):
#     features = []
#     data = epoch.get_data()
#     for i in range(data.shape[1]):
#         channel_data = data[:, i].squeeze()
#         features.append(np.mean(channel_data))
#         features.append(np.std(channel_data))
#         features.append(skew(channel_data))
#         features.append(kurtosis(channel_data))
#     return features

# def extract_wavelet_features(epoch):
#     coeffs = []
#     data = epoch.get_data()
#     for i in range(data.shape[1]):
#         channel_data = data[:, i].squeeze()
#         cA, cD = pywt.dwt(channel_data, 'db4')  # Decompose signal using Daubechies 4 wavelet
#         coeffs.append(cA.mean())
#         coeffs.append(cA.std())
#         coeffs.append(np.abs(cD).mean())  # Take absolute value of cD coefficients to account for negative values
#         coeffs.append(np.abs(cD).std())
#         coeffs.append(skew(np.abs(cD).flatten()))  # Skewness of cD coefficients
#         coeffs.append(kurtosis(np.abs(cD).flatten()))  # Kurtosis of cD coefficients
#         coeffs.append(np.sum(np.square(cA)))  # Energy of wavelet coefficients
#         coeffs.append(np.sum(np.square(cD)))  # Energy of detail coefficients
#         coeffs.append(-np.sum(np.square(cA) * np.log(np.square(cA))))  # Wavelet entropy of approximation coefficients
#         coeffs.append(-np.sum(np.square(cD) * np.log(np.square(cD))))  # Wavelet entropy of detail coefficients
#     return coeffs


# def extract_autocorrelation_features(epoch, max_lag=None):
#     features = []
#     data = epoch.get_data()
#     for i in range(data.shape[1]):
#         channel_data = data[:, i].squeeze()
#         autocorr = correlate(channel_data, channel_data, mode='full')
#         autocorr = autocorr[autocorr.size // 2:]  # Keep only the positive lags

#         if max_lag is not None:
#             autocorr = autocorr[:max_lag]

#         max_lag_at_max_autocorr = np.argmax(autocorr)  # Find the time lag with maximum autocorrelation
#         features.append(max_lag_at_max_autocorr)

#     return features

# def shannon_entropy(channel_data):
#     p_data = np.histogram(channel_data, bins=64, density=True)[0]
#     return entropy(p_data)

# def approximate_entropy(channel_data, order=2, metric='chebyshev'):
#     return ent.approximate_entropy(channel_data, order, metric)

# def sample_entropy(channel_data, order=2, metric='chebyshev'):
#     return ent.sample_entropy(channel_data, order, metric)

# def permutation_entropy(channel_data, order=3, delay=1, normalize=False):
#     return ent.permutation_entropy(channel_data, order, delay, normalize)

# def extract_entropy_features(epoch):
#     features = []
#     data = epoch.get_data()
#     for i in range(data.shape[1]):
#         channel_data = data[:, i].squeeze()
#         features.append(shannon_entropy(channel_data))
#         features.append(approximate_entropy(channel_data))
#         features.append(sample_entropy(channel_data))
#         features.append(permutation_entropy(channel_data))

#     return features


# # def extract_combined_features(all_data):
# #     # Initialize an empty DataFrame to store the combined feature data
# #     combined_feature_df = pd.DataFrame()

# #     # Define a function to extract features for a single data object
# #     def extract_features_for_single_data(data):
# #         psd_features = extract_psd_features(data)
# #         time_domain_features = extract_time_domain_features(data)
# #         wavelet_features = extract_wavelet_features(data)
# #         #autocorrelation_features = extract_autocorrelation_features(data)

# #         # Combine all features into a single list
# #         #combined_features = np.hstack([psd_features, time_domain_features, wavelet_features, autocorrelation_features])
        
# #         entropy_features = extract_entropy_features(data)

# #         # Combine all features into a single list
# #         #combined_features = np.hstack([psd_features, time_domain_features, wavelet_features, autocorrelation_features, entropy_features])
# #         #return combined_features
        
# #         # Combine all features into a single list
# #         combined_features = np.hstack([psd_features, time_domain_features, wavelet_features])
# #         return combined_features
    
# #     # Use Joblib to parallelize the feature extraction
# #     combined_features_list = Parallel(n_jobs=-1)(delayed(extract_features_for_single_data)(all_data[i]) for i in range(len(all_data)))

# #     # Append the combined features as rows in the combined_feature_df dataframe
# #     for combined_features in combined_features_list:
# #         combined_feature_df = combined_feature_df.append(pd.Series(combined_features), ignore_index=True)

# #     return combined_feature_df


# def extract_combined_features(all_data):
#     # Initialize an empty DataFrame to store the combined feature data
#     combined_feature_df = pd.DataFrame()

#     # Define a function to extract features for a single data object
#     def extract_features_for_single_data(data):
#         psd_features = extract_psd_features(data)
#         time_domain_features = extract_time_domain_features(data)
#         wavelet_features = extract_wavelet_features(data)
#         entropy_features = extract_entropy_features(data)

#         # Combine all features into a single list
#         combined_features = np.hstack([psd_features, time_domain_features, wavelet_features])

#         return combined_features
    
#     # Use Joblib to parallelize the feature extraction and tqdm to show the progress
#     combined_features_list = Parallel(n_jobs=-1)(delayed(extract_features_for_single_data)(all_data[i]) for i in tqdm(range(len(all_data)), desc="Extracting combined features"))

#     # Append the combined features as rows in the combined_feature_df dataframe
#     for combined_features in combined_features_list:
#         combined_feature_df = combined_feature_df.append(pd.Series(combined_features), ignore_index=True)

#     return combined_feature_df


# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from mne import Epochs
from joblib import Parallel, delayed
from tqdm import tqdm

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
    return coeffs

def extract_combined_features(all_data):
    # Initialize an empty DataFrame to store the combined feature data
    combined_feature_df = pd.DataFrame()

    # Define a function to extract features for a single data object
    def extract_features_for_single_data(data):
        psd_features = extract_psd_features(data)
        time_domain_features = extract_time_domain_features(data)
        wavelet_features = extract_wavelet_features(data)

        # Combine all features into a single list
        combined_features = np.hstack([psd_features, time_domain_features, wavelet_features])
        return combined_features

    # Use Joblib to parallelize the feature extraction
    #combined_features_list = Parallel(n_jobs=-1)(delayed(extract_features_for_single_data)(all_data[i]) for i in range(len(all_data)))
       # Use Joblib to parallelize the feature extraction and tqdm to show the progress
    combined_features_list = Parallel(n_jobs=-1)(delayed(extract_features_for_single_data)(all_data[i]) for i in tqdm(range(len(all_data)), desc="Extracting combined features"))

    # Append the combined features as rows in the combined_feature_df dataframe
    for combined_features in combined_features_list:
        combined_feature_df = combined_feature_df.append(pd.Series(combined_features), ignore_index=True)

    return combined_feature_df

