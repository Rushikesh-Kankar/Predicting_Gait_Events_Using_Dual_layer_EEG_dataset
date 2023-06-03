#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.io
import numpy as np
import pandas as pd
import mne

def preprocess_eeg_data(subject_ids, file_path, event_id, save_dir, use_78_channels=False):
    # Create an empty list to store the preprocessed EEG data for each subject
    eeg_data = []
    
    for sub_id in subject_ids:
        # Load the data from the .set file
        full_path = file_path.format(sub_id=sub_id)
        if 'allwalk_EEG.set' in full_path:
            
            if use_78_channels:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                # extract good channels
                mat_file = scipy.io.loadmat(f'DATA/EEG_ICA_STRUCT/PTW{sub_id}_allwalk_EEG_ICA_STRUCT_rejbadchannels_diverse_incr_comps.mat')
                good_channels = mat_file['EEG_ICA_STRUCT']['good_chans'][0][0][0]
                # randomly select 78 good channels
                selected_channels = np.random.choice(good_channels, size=78, replace=False)
                raw.pick_channels([raw.ch_names[i] for i in selected_channels - 1])
                # rename the channels to the same names
                channel_names = ['ch' + str(i+1) for i in range(len(raw.ch_names))]
                raw.rename_channels(dict(zip(raw.ch_names, channel_names)))
                events = mne.events_from_annotations(raw)
                events1 = events[:1][0]
                epochs = mne.Epochs(raw, events1, event_id=event_id, tmin=-0.05, tmax=0.05, event_repeated='merge')
                eeg_data.append(epochs)
                
            else:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                events = mne.events_from_annotations(raw)
                events1 = events[:1][0]
                epochs = mne.Epochs(raw, events1, event_id=event_id, tmin=-0.05, tmax=0.05, event_repeated='merge')
                eeg_data.append(epochs)
    
        elif 'allwalk_artifact.set' in full_path:
            if use_78_channels:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                # randomly select 78 channels
                selected_channels = np.random.choice(raw.ch_names, size=78, replace=False)
                raw.pick_channels(selected_channels)
                # rename the channels to the same names
                channel_names = ['ch' + str(i+1) for i in range(len(raw.ch_names))]
                raw.rename_channels(dict(zip(raw.ch_names, channel_names)))
                events = mne.events_from_annotations(raw)
                events1 = events[:1][0]
                epochs = mne.Epochs(raw, events1, event_id=event_id, tmin=-0.05, tmax=0.05, event_repeated='merge')
                eeg_data.append(epochs)

            else:    
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                events = mne.events_from_annotations(raw)
                events1 = events[:1][0]
                epochs = mne.Epochs(raw, events1, event_id=event_id, tmin=-0.05, tmax=0.05, event_repeated='merge')
                eeg_data.append(epochs)
                
                
        else:
            print(f'File path for subject {sub_id} is invalid')
    
    # Concatenate the data from all subjects
    all_data = mne.concatenate_epochs(eeg_data)
    
    # Save the epoched data for each event type
    for event_id1 in event_id:
        save_path = f'{save_dir}/my_epochs_{event_id1}-epo.fif'
        all_data[event_id1].save(save_path, overwrite=True)
    
    return all_data

