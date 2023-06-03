#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import mne
import pandas as pd

def extract_window_data(file_path, event_dict, first_event, last_event, window_times, save_dir=None):
    # Load the EEG data from the .set file into an MNE raw object
    raw = mne.io.read_raw_eeglab(file_path)

    # Convert annotations to events
    events, event_id = mne.events_from_annotations(raw, event_id=event_dict, chunk_duration=None)

    # Create a list to store event information
    event_info = []

    # Loop over each event
    for ev in events:
        # Get the event name and time
        event_name = list(event_dict.keys())[list(event_dict.values()).index(ev[2])]
        event_time = raw.times[ev[0]]
        sample_number = ev[0]
        # Add event information to the list
        event_info.append([event_name, event_time, sample_number])

    # Save the event information to a CSV file
    if save_dir:
        with open(f"{save_dir}/event_info.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Event Name', 'Time', 'Sample Number'])
            for info in event_info:
                writer.writerow(info)

    # Extract data for each window
    window_data = []
    for start_time, end_time in window_times:
        # Extract data in a specific time window
        window_data_df = pd.DataFrame(event_info, columns=['Event Name', 'Time', 'Sample Number'])
        window_data_df = window_data_df[(window_data_df['Time'] >= start_time) & (window_data_df['Time'] <= end_time)]

        # Find the index of the first occurrence of the first event
        first_event_index = window_data_df[window_data_df['Event Name'] == first_event].index[0]

        # Find the index of the last occurrence of the second event
        last_event_index = window_data_df[window_data_df['Event Name'] == last_event].index[-1]

        # Extract data between the two indices
        final_data = window_data_df.loc[first_event_index:last_event_index]

        # Append window data to the list
        window_data.append(final_data)

        # Save window data to a CSV file
        if save_dir:
            file_name = f"{start_time}-{end_time}.csv"
            final_data = final_data.loc[first_event_index:last_event_index]
            final_data.to_csv(f"{save_dir}/{file_name}", index=False)

    return window_data

