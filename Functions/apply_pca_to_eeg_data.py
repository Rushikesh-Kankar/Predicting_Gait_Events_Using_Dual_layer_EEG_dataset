#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def apply_pca_to_eeg_data(all_data):
    # Get the EEG data
    eeg_data = all_data.get_data()

    # The shape of eeg_data is (n_epochs, n_channels, n_times)
    # We want to reshape it to (n_epochs, n_channels * n_times)
    n_epochs, n_channels, n_times = eeg_data.shape
    eeg_data_reshaped = eeg_data.reshape((n_epochs, n_channels * n_times))

    # Standardize the data
    scaler = StandardScaler()
    eeg_data_standardized = scaler.fit_transform(eeg_data_reshaped)

    # Apply PCA and keep enough components to explain 95% of variance
    pca = PCA(n_components=10)
    eeg_data_pca = pca.fit_transform(eeg_data_standardized)

    # Print the number of components
    n_components = eeg_data_pca.shape[1]
    print(f"Number of components that explain 95% of variance: {n_components}")

    # Plot the cumulative sum of explained variance ratio
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(explained_variance)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Get event labels
    events = all_data.events[:, 2]  # the event codes

    # Reverse the dictionary
    event_id = all_data.event_id  # a dictionary mapping event codes to event names
    event_id = {v: k for k, v in event_id.items()}

    # Map event codes to event names
    event_names = [event_id[event] for event in events]

    # Create a DataFrame with the first two principal components and the labels
    data = {'PC1': eeg_data_pca[:, 0], 'PC2': eeg_data_pca[:, 1], 'label': event_names}
    df = pd.DataFrame(data)

    # Create a scatter plot with colors
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='label', palette='Set2', alpha=0.7)
    plt.title('Scatter plot of the first two principal components colored by label')
    plt.show()

    return eeg_data_pca, events

