#!/usr/bin/env python
# coding: utf-8

# In[3]:


import mne

def load_data(event_id, file_path):
    all_epochs = []
    for event in event_id:
        epochs = mne.read_epochs(f'{file_path}/my_epochs_{event}-epo.fif')
        all_epochs.append(epochs)
    all_data = mne.concatenate_epochs(all_epochs)
    return all_data

