# EEG-based Gait Event Prediction Project

This project is designed to process EEG data and predict gait events. It makes use of machine learning techniques and various neural network architectures for the prediction process. The entire workflow of the project is bundled in a Jupyter notebook called `EEG_gait_prediction_pipeline.ipynb`.

## Project Structure

### EEG_gait_prediction_pipeline.ipynb

This is the main Jupyter notebook, which coordinates the overall workflow. It imports and utilizes the functions from various modules:

1. **preprocess_eeg_data.ipynb** - Concatenates the epochs data for multiple subjects into a single file for training the models.
2. **Read_data.ipynb** - Loads the preprocessed data from the first module and prepares it for model training.
3. **Feature_Extraction_Parallel/Feature_extraction_all_channels_Parallel.py** - Extracts relevant features from the EEG signal to reduce the dimensionality of the data and capture more informative features.
4. **Training_Models2.ipynb** - Trains different machine learning models such as Random Forest, Decision Trees, Linear Regression, Support Vector Machines (SVM), and Neural Networks using the preprocessed data and checks their accuracy.
5. **Extract_Window_Data.ipynb** - Extracts the required windowed data for making predictions on raw data.
6. **Final_Predictions_on_raw_data.ipynb** - Makes predictions on new raw data using the trained machine learning models. The predicted events and their times are saved to a CSV file.

## Requirements

* Python 3.7 or above
* Libraries: MNE, pandas, scikit-learn, numpy, joblib, pickle, concurrent.futures

## Installation

1. Clone the repository.
2. Ensure that you have the required Python version and all necessary libraries installed.
3. Navigate to the project directory.

## Usage

Run `EEG_gait_prediction_pipeline.ipynb` in a Jupyter notebook.

## Results

The project has been run on four different datasets. The performance of the models was poorest for the scalp layer with only good channels, suggesting that the bad channels contained necessary information about motion artifacts, which enabled the models to accurately predict gait events. In contrast, the scalp layer with only good channels lacked this information, leading to suboptimal model performance.

## Future Work

The next steps in this project include reducing the epoch duration, extracting more relevant features, training the model on ICA data, hyperparameter tuning, and incorporating neural networks for prediction.

## Contributors

* Rushikesh Kankar
* ...

## License



## Acknowledgments

## Acknowledgments

I would like to express my sincere gratitude to Dr. Helen Huang for her invaluable guidance and support throughout this project. Her expertise and insights have greatly enriched my work.



**Please note that these results are preliminary and may change as the project progresses. For the most up-to-date results, please check back periodically.**

This document is last updated on 10th May 2023.
