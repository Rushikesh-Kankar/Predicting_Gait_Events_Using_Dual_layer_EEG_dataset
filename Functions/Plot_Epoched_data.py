#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
import matplotlib.pyplot as plt
from ipywidgets import interact

def plot_eeg_features(raw_file_path, epochs, z, plots_to_print=None):
    """
    Plots various features of the EEG data for each channel up to the indices in z.

    Parameters:
    raw_file_path (str): The path to the raw EEG data file.
    epochs (mne.Epochs): The MNE Epochs object containing the EEG data.
    z (list of int): The list of indices of the last channels to plot.
    plots_to_print (list of str, optional): The list of plots to display. Defaults to None.

    Returns:
    None
    """
    if plots_to_print is None:
        plots_to_print = ['psd', 'image', 'event_plot', 'event_average', 'average', 'info', 'plot_epochs', 'interactive_plot', 'peak_latency']

    raw = mne.io.read_raw_eeglab(raw_file_path, preload=True)
    
    # Plot PSD for each epoch
    if 'psd' in plots_to_print:
        for i in z:
            fig = epochs[i].plot_psd()
            fig.canvas.manager.set_window_title(f"PSD plot for channel {i}")
            plt.title(f"PSD plot for epoch {i}")

    # Plot image for each epoch
    if 'image' in plots_to_print:
        for i in z:
            figs = epochs[i].plot_image()
            for j, fig in enumerate(figs):
                fig.canvas.manager.set_window_title(f"Image plot for channel {i}, figure {j+1}")
                plt.title(f"Image plot for epoch {i}")

    # Plot averaged data for each epoch
    if 'average' in plots_to_print:
        for i in z:
            fig = epochs[i].average().plot()
            fig.canvas.manager.set_window_title(f"Average plot for channel {i}")
            plt.title(f"Average plot for epoch {i}")

    # Plot info for each epoch
    if 'info' in plots_to_print:
        for i in z:
            print(f"Info for epoch {i}:")
            print(epochs[i].info)

    # Plot of epochs
    if 'plot_epochs' in plots_to_print:
        for i in z:
            print(f"Plot for Epoch{i}:")
            mne.viz.plot_epochs(epochs[i], scalings='auto')

    # Define a function to plot the EEG data with interactive time slider
    if 'interactive_plot' in plots_to_print:
        @interact(t=(0, raw.times[-1], 0.1))
        def plot_eeg(t):
            raw.plot(start=t, duration=5, scalings='auto')

    # Get the peak latency for each event
    if 'peak_latency' in plots_to_print:
        for i in range(4):
            start_index = i * len(epochs) // 4
            end_index = (i + 1) * len(epochs) // 4
            peak_latency = epochs[start_index:end_index].average().get_peak()[1]
            print(f"Peak latency for event {start_index} to {end_index}: {peak_latency} s")

