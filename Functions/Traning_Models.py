#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

def train_models_and_save_raw_data(all_data, save_dir, random_seed=42):
    # Get the data and labels
    X = all_data.get_data()
    y = all_data.events[:, -1].astype(int)
    X = X.reshape(X.shape[0], -1)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, train_size=0.3, random_state=random_seed
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and save models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        #"SVM": SVC(),
        #"Logistic Regression": LogisticRegression(),
        #"SVM with SGD": SGDClassifier(loss='hinge', alpha=0.0001, random_state=42, max_iter=1000, tol=1e-3)
    }
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{model_name} Accuracy: {accuracy}")

        # Print and write the classification report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        report_filename = os.path.join(save_dir, f"{model_name}_report.txt")
        try:
            with open(report_filename, "w") as f:
                f.write(report)
        except:
            print(f"Failed to write report file for {model_name}")

        # Print and write the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        cm_filename = os.path.join(save_dir, f"{model_name}_confusion_matrix.txt")
        try:
            with open(cm_filename, "w") as f:
                f.write(str(cm))
        except:
            print(f"Failed to write confusion matrix file for {model_name}")

        # Save the model to a file
        model_filename = os.path.join(save_dir, f"{model_name}.pkl")
        try:
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
        except:
            print(f"Failed to write model file for {model_name}")



def train_models_and_save_feature_extracted_data(X,y, save_dir, random_seed=42):
    # Get the data and labels
  
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, train_size=0.3, random_state=random_seed
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and save models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        #"SVM": SVC(),
        #"Logistic Regression": LogisticRegression(),
        #"SVM with SGD": SGDClassifier(loss='hinge', alpha=0.0001, random_state=42, max_iter=1000, tol=1e-3)
    }
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{model_name} Accuracy: {accuracy}")

        # Print and write the classification report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        report_filename = os.path.join(save_dir, f"{model_name}_report.txt")
        try:
            with open(report_filename, "w") as f:
                f.write(report)
        except:
            print(f"Failed to write report file for {model_name}")

        # Print and write the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        cm_filename = os.path.join(save_dir, f"{model_name}_confusion_matrix.txt")
        try:
            with open(cm_filename, "w") as f:
                f.write(str(cm))
        except:
            print(f"Failed to write confusion matrix file for {model_name}")

        # Save the model to a file
        model_filename = os.path.join(save_dir, f"{model_name}.pkl")
        try:
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
        except:
            print(f"Failed to write model file for {model_name}")

            

def train_models_and_save_pca_data(eeg_data_pca, event_names, save_dir, random_seed=42):
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        eeg_data_pca, event_names, test_size=0.7, train_size=0.3, random_state=random_seed
    )

    # Train and save models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    }
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{model_name} Accuracy: {accuracy}")

        # Print and write the classification report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        report_filename = os.path.join(save_dir, f"{model_name}_report.txt")
        try:
            with open(report_filename, "w") as f:
                f.write(report)
        except:
            print(f"Failed to write report file for {model_name}")

        # Print and write the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        cm_filename = os.path.join(save_dir, f"{model_name}_confusion_matrix.txt")
        try:
            with open(cm_filename, "w") as f:
                f.write(str(cm))
        except:
            print(f"Failed to write confusion matrix file for {model_name}")

        # Save the model to a file
        model_filename = os.path.join(save_dir, f"{model_name}.pkl")
        try:
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
        except:
            print(f"Failed to write model file for {model_name}")

