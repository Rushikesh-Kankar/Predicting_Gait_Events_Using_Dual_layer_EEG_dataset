#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def lstm_model(data, save_dir, test_size=0.2, random_state=42, epochs=100, batch_size=32):
    # Prepare the data for training
    X = data.get_data()
    y = data.events[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Normalize the data
    scaler = StandardScaler()
    for i in range(X_train.shape[1]):
        X_train[:, i, :] = scaler.fit_transform(X_train[:, i, :])
        X_test[:, i, :] = scaler.transform(X_test[:, i, :])

    # Reshape the data to fit the LSTM input shape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('Test accuracy:', test_acc)

    # Save evaluation metrics and model
    model_name = 'LSTM'

    # Predict the classes
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Print and write the classification report
    report = classification_report(y_test, y_pred)
    print(report)
    report_filename = os.path.join(save_dir, f"{model_name}_report.txt")
    with open(report_filename, "w") as f:
        f.write(report)

    # Print and write the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cm_filename = os.path.join(save_dir, f"{model_name}_confusion_matrix.txt")
    with open(cm_filename, "w") as f:
        f.write(str(cm))

    # Save the model to a file
    model_filename = os.path.join(save_dir, f"{model_name}.h5")
    model.save(model_filename)
    
    return model, history

