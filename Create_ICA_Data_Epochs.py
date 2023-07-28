#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import mne
import numpy as np
import scipy.io
import pandas as pd
import random

def find_continuous_no_event_epochs(events, raw, num_ulhs):
    event_id_no_event = {'no_event': 0}

    event_df = pd.DataFrame(events[0], columns=['Sample Number', 'Offset', 'Event ID'])
    event_name_dict = events[1]  
    reverse_dict = {v: k for k, v in event_name_dict.items()}
    event_df['Event Name'] = event_df['Event ID'].map(reverse_dict)

    unperturbed_frame = event_df.loc[event_df['Event Name'].isin(['ULHS', 'URTO', 'URHS', 'ULTO','ULP_On'])] 

    # Sort by Sample Number
    unperturbed_frame = unperturbed_frame.sort_values('Sample Number')

    # Get the sample numbers
    sample_numbers = unperturbed_frame['Sample Number'].values

    middle_samples = []
    # For each pair of consecutive sample numbers in the unperturbed frame
    for i in range(len(sample_numbers) - 1):
        # If the difference is larger than 53
        if sample_numbers[i+1] - sample_numbers[i] > 103:
            # Get the middle sample number
            middle_sample = (sample_numbers[i] + sample_numbers[i+1]) // 2
            middle_samples.append(middle_sample)

    # Randomly select middle samples
    if len(middle_samples) > num_ulhs:
        middle_samples = random.sample(middle_samples, num_ulhs)

    # Create epochs for the selected middle samples
    continuous_no_event_epochs_list = []
    for middle_sample in middle_samples:
        new_events = np.array([[middle_sample, 0, 0]])
        continuous_no_event_epochs_list.append(
            mne.Epochs(raw, new_events, event_id=event_id_no_event, tmin=-0.1, tmax=0.1, event_repeated='merge', baseline=(None,0))
        )

    return continuous_no_event_epochs_list


def load_ica_data(subject_code):
    ica_file = f'DATA/EEG_ICA_STRUCT/PTW{subject_code}_allwalk_EEG_ICA_STRUCT_rejbadchannels_diverse_incr_comps.mat'
    ica_data = scipy.io.loadmat(ica_file)
    icaweights = ica_data['EEG_ICA_STRUCT']['weights'][0][0]
    icasphere = ica_data['EEG_ICA_STRUCT']['sphere'][0][0]
    good_channels = ica_data['EEG_ICA_STRUCT']['good_chans'][0][0][0] - 1  # Subtract 1 to correct the indices
    return icaweights, icasphere, good_channels

# def epoch_ica_data(ica_data, events, event_id, tmin=-0.05, tmax=0.05):
#     info = mne.create_info(ch_names=[f'IC{i+1}' for i in range(ica_data.shape[0])], sfreq=raw.info['sfreq'], ch_types='eeg')
#     ica_raw = mne.io.RawArray(ica_data, info)
#     epochs = mne.Epochs(ica_raw, events, event_id, tmin=tmin, tmax=tmax)
#     return epochs

def epoch_ica_data(ica_data, events, event_id, tmin=-0.1, tmax=0.1):
    info = mne.create_info(ch_names=[f'{i+1}' for i in range(ica_data.shape[0])], sfreq=raw.info['sfreq'], ch_types='eeg')
    ica_raw = mne.io.RawArray(ica_data, info)
 #   print(events)
    epochs = mne.Epochs(ica_raw, events, event_id, tmin=tmin, tmax=tmax)
    return epochs


def apply_ica_weights(raw, icaweights, icasphereou, good_channels, num_ics=78):
    # Get the channel names for the good channels
    good_channel_names = [raw.ch_names[i] for i in good_channels]

    # Get the data from the good channels only
    good_channel_data = raw.copy().pick_channels(good_channel_names).get_data()

    # Compute the unmixing matrix using the provided ICA weights and sphere matrix
    unmixing_matrix = np.dot(icaweights, icasphere)

    # Select the top 78 components based on the sum of the absolute weights
    component_weights = np.sum(np.abs(unmixing_matrix), axis=1)
    top_ics_indices = np.argsort(component_weights)[-num_ics:]

    # Apply the unmixing matrix for the top 78 ICA components to the good channel data
    ica_data = np.dot(unmixing_matrix[top_ics_indices, :], good_channel_data)

    return ica_data


subject_ids = range(1, 15)
event_id = {'ULHS': 11, 'ULTO': 13, 'URHS': 14, 'URTO': 15}
save_dir = 'DATA/Event_Epoched_Data_ICA_78'  # Specify your desired save directory

# Make sure the save directory exists
os.makedirs(save_dir, exist_ok=True)


all_epochs = {event: [] for event in event_id}
eeg_data=[]
continuous_no_event_epochs_list_all=[]


for subject in subject_ids:
    subject_code = f'{subject:02d}'
    set_file = f'DATA/Data_Inner_Layer/PTW{subject_code}_allwalk_EEG.set'

    # Load the .set file
    raw = mne.io.read_raw_eeglab(set_file, preload=True)

    # Load the ICA data
    icaweights, icasphere, good_channels = load_ica_data(subject_code)

    # Apply the ICA weights to the raw data using the provided ICA weights and sphere matrix
    ica_data = apply_ica_weights(raw, icaweights, icasphere, good_channels)

    # Get the events from the raw data
    events=mne.events_from_annotations(raw)#, event_id_list = mne.events_from_annotations(raw)
    

    # Epoch the ICA activations around the events
    epochs = epoch_ica_data(ica_data, events[:1][0], event_id)

#     for event in event_id:
#         all_epochs[event].append(epochs[event])

    eeg_data.append(epochs)
    
    # Get the events array
    events_array = epochs.events
    # Get the event_id for 'ULHS'
    ulhs_id = event_id['ULHS']
    # Count the number of 'ULHS' epochs
    num_ulhs = np.sum(events_array[:, 2] == ulhs_id)

    # Create ica_raw
    info = mne.create_info(ch_names=[f'{i+1}' for i in range(ica_data.shape[0])], sfreq=raw.info['sfreq'], ch_types='eeg')
    ica_raw = mne.io.RawArray(ica_data, info)    
    
    # Find continuous no-event epochs
    continuous_no_event_epochs_list = find_continuous_no_event_epochs(events, ica_raw,num_ulhs)
    
    continuous_no_event_epochs_list_all += continuous_no_event_epochs_list
    
all_data_event = mne.concatenate_epochs(eeg_data)


# Add the selected no-event epochs to the all epochs list
all_epochs_list = eeg_data + continuous_no_event_epochs_list_all

# Concatenate the event-related and selected no-event epochs
all_data = mne.concatenate_epochs(all_epochs_list)

# Add no_event to the event_id dictionary
event_id['no_event'] = 0

    

    
# Concatenate and save the epoched data for each event type
for event in event_id:
    #concatenated_epochs = mne.concatenate_epochs(all_epochs[event])
    save_path = f'{save_dir}/my_epochs_{event}-epo.fif'
    all_data[event].save(save_path, overwrite=True)

