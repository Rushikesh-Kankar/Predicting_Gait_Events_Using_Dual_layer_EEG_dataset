#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.io
import numpy as np
import pandas as pd
import mne
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def apply_pca_to_raw(raw, max_components):
    data = raw.get_data().T
    pca = PCA(n_components=max_components)
    transformed_data = pca.fit_transform(data)
    
    # Create new channel names for PCA components
    new_ch_names = [f'PCA{str(i+1)}' for i in range(max_components)]
    
    # Create a new raw object with the transformed data
    info = mne.create_info(ch_names=new_ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    raw_pca = mne.io.RawArray(transformed_data.T, info)
    
    return raw_pca


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
        if sample_numbers[i+1] - sample_numbers[i] > 53:
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
            mne.Epochs(raw, new_events, event_id=event_id_no_event, tmin=-0.05, tmax=0.05, event_repeated='merge', baseline=(None,0))
        )

    return continuous_no_event_epochs_list


def get_max_pca_components(subject_ids, file_path, use_78_channels=False):
    max_components = 0
    pc_count_dict = {}

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
                np.random.seed(42) 
                selected_channels = np.random.choice(good_channels, size=78, replace=False)
                # Save the selected channel names to a file
                with open(f'{save_dir}/selected_channel_names_{sub_id}.txt', 'w') as f:
                    for channel in selected_channels:
                        f.write(f'{channel}\n')
                raw.pick_channels([raw.ch_names[i] for i in selected_channels - 1])
                # rename the channels to the same names
                channel_names = ['ch' + str(i+1) for i in range(len(raw.ch_names))]
                raw.rename_channels(dict(zip(raw.ch_names, channel_names)))
                
            else:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)

            # Calculate PCA components after preprocessing
            data = raw.get_data().T
            pca = PCA(n_components=0.95)
            pca.fit(data)
            
            explained_variance = pca.explained_variance_ratio_

#             for i, variance in enumerate(explained_variance):
#                 print(f"Component {i+1}: {variance:.5f} variance explained")
            
#             cumulative_variance_at_20 = sum(pca.explained_variance_ratio_[:20])
#             print(f"Cumulative variance at component 20: {cumulative_variance_at_20:.5f}")


            max_components = max(max_components, pca.n_components_)
            pc_count_dict[sub_id] = pca.n_components_
            

            
            
            # Plotting the cumulative sum of explained variance ratio
            explained_variance = np.cumsum(pca.explained_variance_ratio_)

            plt.figure(figsize=(10, 6))
            
            plt.plot(explained_variance)
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.title(f'Subject {sub_id}: Cumulative Explained Variance by Components')
            plt.tight_layout()
            ax = plt.gca()  # Get the current axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.savefig(f'{save_dir}/{sub_id}.png')
            plt.close()
            
        elif 'allwalk_artifact.set' in full_path:
            
            if use_78_channels:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                np.random.seed(42)
                # randomly select 78 channels
                selected_channels = np.random.choice(raw.ch_names, size=78, replace=False)
                raw.pick_channels(selected_channels)
                # Save the selected channel names to a file
                with open(f'{save_dir}/selected_channel_names_{sub_id}.txt', 'w') as f:
                    for channel in selected_channels:
                        f.write(f'{channel}\n')
                # rename the channels to the same names
                channel_names = ['ch' + str(i+1) for i in range(len(raw.ch_names))]
                raw.rename_channels(dict(zip(raw.ch_names, channel_names)))


            else:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
            
          # Calculate PCA components after preprocessing
            data = raw.get_data().T
            pca = PCA(n_components=0.95)
            pca.fit(data)
            
            explained_variance = pca.explained_variance_ratio_

#             for i, variance in enumerate(explained_variance):
#                 print(f"Component {i+1}: {variance:.5f} variance explained")
            
#             cumulative_variance_at_20 = sum(pca.explained_variance_ratio_[:20])
#             print(f"Cumulative variance at component 20: {cumulative_variance_at_20:.5f}")


            max_components = max(max_components, pca.n_components_)
            pc_count_dict[sub_id] = pca.n_components_
            

            
            
            # Plotting the cumulative sum of explained variance ratio
            explained_variance = np.cumsum(pca.explained_variance_ratio_)

            plt.figure(figsize=(10, 6))
            
            plt.plot(explained_variance)
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.title(f'Subject {sub_id}: Cumulative Explained Variance by Components')
            plt.tight_layout()
            ax = plt.gca()  # Get the current axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.savefig(f'{save_dir}/{sub_id}.png')
            plt.close()
            
        else:
            print(f'File path for subject {sub_id} is invalid')

    
    df = pd.DataFrame(list(pc_count_dict.items()), columns=['Subject ID', 'Number of PCs'])
    df.to_csv('num_pcs.csv', index=False)

    return max_components


def preprocess_eeg_data(subject_ids, file_path, event_id, save_dir, use_78_channels=False,use_pca_for_epoching=False):
    # Create an empty list to store the preprocessed EEG data for each subject
    eeg_data = []
    continuous_no_event_epochs_list_all=[]
    pc_count_dict = {}
    max_components = get_max_pca_components(subject_ids, file_path, use_78_channels)
    
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
                np.random.seed(42)
                selected_channels = np.random.choice(good_channels, size=78, replace=False)
                # Save the selected channel names to a file
                with open(f'{save_dir}/selected_channel_names_{sub_id}.txt', 'w') as f:
                    for channel in selected_channels:
                        f.write(f'{channel}\n')
                raw.pick_channels([raw.ch_names[i] for i in selected_channels - 1])
                # rename the channels to the same names
                channel_names = ['ch' + str(i+1) for i in range(len(raw.ch_names))]
                raw.rename_channels(dict(zip(raw.ch_names, channel_names)))
                

            else:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
            
            events = mne.events_from_annotations(raw)
            events1 = events[:1][0]

            
            if(use_pca_for_epoching==True):
                
                raw = apply_pca_to_raw(raw, max_components)
                
            # Create the Epochs object with preload=True
            epochs = mne.Epochs(raw, events1, event_id, tmin=-0.05, tmax=0.05, preload=True, event_repeated='merge')
            # Get the events array
            events_array = epochs.events
            # Get the event_id for 'ULHS'
            ulhs_id = event_id['ULHS']
            # Count the number of 'ULHS' epochs
            num_ulhs = np.sum(events_array[:, 2] == ulhs_id)
            #Find continuous no-event epochs
            eeg_data.append(epochs)
            continuous_no_event_epochs_list = find_continuous_no_event_epochs(events,raw,num_ulhs)
            continuous_no_event_epochs_list_all += continuous_no_event_epochs_list
            
                
        elif 'allwalk_artifact.set' in full_path:
            if use_78_channels:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                # randomly select 78 channels
                selected_channels = np.random.choice(raw.ch_names, size=78, replace=False)
                raw.pick_channels(selected_channels)
                # Save the selected channel names to a file
                with open(f'{save_dir}/selected_channel_names_{sub_id}.txt', 'w') as f:
                    for channel in selected_channels:
                        f.write(f'{channel}\n')
                # rename the channels to the same names
                channel_names = ['ch' + str(i+1) for i in range(len(raw.ch_names))]
                raw.rename_channels(dict(zip(raw.ch_names, channel_names)))


            else:
                raw = mne.io.read_raw_eeglab(full_path, preload=True)
                
            events = mne.events_from_annotations(raw)
            events1 = events[:1][0]
            
            if(use_pca_for_epoching==True):
                raw = apply_pca_to_raw(raw, max_components)
                
            epochs = mne.Epochs(raw, events1, event_id=event_id, tmin=-0.05, tmax=0.05, event_repeated='merge')
            # Get the events array
            events_array = epochs.events
            # Get the event_id for 'ULHS'
            ulhs_id = event_id['ULHS']
            # Count the number of 'ULHS' epochs
            num_ulhs = np.sum(events_array[:, 2] == ulhs_id)
            #Find continuous no-event epochs
            eeg_data.append(epochs)
            continuous_no_event_epochs_list = find_continuous_no_event_epochs(events,raw,num_ulhs)
            continuous_no_event_epochs_list_all += continuous_no_event_epochs_list



        else:
            print(f'File path for subject {sub_id} is invalid')
   

    all_data_event = mne.concatenate_epochs(eeg_data)

    
    # Add the selected no-event epochs to the all epochs list
    all_epochs_list = eeg_data + continuous_no_event_epochs_list_all

    # Concatenate the event-related and selected no-event epochs
    all_data = mne.concatenate_epochs(all_epochs_list)

    # Add no_event to the event_id dictionary
    event_id['no_event'] = 0

    # Save the epoched data for each event type including the no-event epochs
    for event_id1 in event_id:
        save_path = f'{save_dir}/my_epochs_{event_id1}-epo.fif'
        all_data[event_id1].save(save_path, overwrite=True)

    return all_data

