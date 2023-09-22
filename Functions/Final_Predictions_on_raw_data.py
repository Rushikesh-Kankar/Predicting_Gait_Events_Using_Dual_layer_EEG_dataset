#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from joblib import load
from concurrent.futures import ThreadPoolExecutor
import mne
import os

def predict_events1(raw_data, model_path, window_size=0.103515625, step_size=0.001953125, t_min=None, t_max=None, threshold=0.2, n_jobs=14):
    """
    Predicts the events for a given raw data using a trained machine learning model.

    Parameters:
    raw_data (mne.io.Raw): The raw data to predict events for.
    model_path (str): The path to the trained machine learning model file.
    window_size (float): The size of the sliding window used for predictions (in seconds). Default is 0.201171875.
    step_size (float): The step size used for sliding the window (in seconds). Default is 0.001953125.
    t_min (float): The minimum time value to start the sliding window. Default is None (use the beginning of the data).
    t_max (float): The maximum time value to end the sliding window. Default is None (use the end of the data).
    threshold (float): The probability threshold used for classifying events. Default is 0.5.
    n_jobs (int): The number of parallel jobs to run. Default is 1 (no parallelization).

    Returns:
    event_preds_times (numpy.ndarray): An array containing the predicted events and their corresponding times.
    """
    # Load the trained machine learning model
    model = load(model_path)
   # print(model)
    # Get the time limits of the data
    if t_min is None:
        t_min = raw_data.times[0]
        print(t_min)
    if t_max is None:
        t_max = raw_data.times[-1]
        print(t_max)
        
    # Set up an empty list to store the predicted events and their times
    event_preds = []
    event_times = []

    # Set up the thread pool executor
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Define the predict function to run in parallel
        def predict(t_start):
            t_end = t_start + window_size

            # Extract the data for the current window
            data = raw_data.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False).get_data()

            # Reshape the data to match the shape expected by the model
            data = data.reshape(1, -1)

            # Apply the trained machine learning model to predict the probability of each trial belonging to each class
            y_pred_proba = model.predict_proba(data)
            print(y_pred_proba)

            # Assign each trial to the most likely event using a threshold on the predicted probabilities
            y_pred = np.argmax(y_pred_proba, axis=1) if np.max(y_pred_proba) > threshold else -1

            # Return the predicted event for the current window
            if y_pred != -1:
                event_pred = y_pred[0]
                event_time = (t_start + t_end) / 2
                return (event_pred, event_time)
            else:
                return None

        # Loop over the sliding windows in parallel
        futures = []
        for t_start in np.arange(t_min, t_max - window_size, step_size):
            futures.append(executor.submit(predict, t_start))

        # Retrieve the predicted events and their times from the completed futures
        for future in futures:
            result = future.result()
            if result is not None:
                event_pred = result[0]
                event_time = result[1]
                event_str = f"Predicted event: {event_pred}, time: {event_time:.3f}"
                print(event_str)
                event_preds.append(event_pred)
                event_times.append(event_time)

        # Convert the list of predicted events and their times to numpy arrays
        event_preds = np.array(event_preds).astype(int)
        event_times = np.array(event_times)

        # Combine the predicted events and their times into a single DataFrame
        df = pd.DataFrame({'event': event_preds, 'time': event_times})
        
        # Get the base file name of the raw data
        raw_file_name = os.path.basename(raw_data.filenames[0])

        # Construct the output file name
        output_file_name = raw_file_name.split('.')[0] + '_predicted_events.csv'
        output_dir = 'Final_Prediction'
        # Save the predicted events to a CSV file
        df.to_csv(output_dir + '/' + output_file_name, index=False)

#         output_dir = 'Final_Prediction'
#         output_file = 'predicted_events.csv'
#         df.to_csv(os.path.join(output_dir, output_file), index=False)

        return df



def save_event_times(df, output_dir):
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # perform the calculations and create the result dataframe
    df['event_group'] = (df['event'] != df['event'].shift()).cumsum()
    grouped = df.groupby('event_group')
    result = grouped.agg({'event': 'first', 'time': ['min', 'max']})
    result.columns = ['event', 'start_time', 'end_time']
    result['time_occured'] = (result['start_time'] + result['end_time']) / 2
    result = result[['event', 'time_occured']]
    result.columns = ['event', 'time_occured']

    # save the result to a CSV file in the output directory
    output_path = os.path.join(output_dir, 'event_times.csv')
    result.to_csv(output_path, index=False)

