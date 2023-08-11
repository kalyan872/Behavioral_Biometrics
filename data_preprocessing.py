import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/keystrokes (1).csv')

# Extract relevant features
features = data[['Hold Times', 'Flight Times', 'Press/Release Timings', 'Key Combinations']]
target = data['Email']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Function to preprocess features
def preprocess_features(data):
    # Convert string representation of lists to actual lists
    data['Hold Times'] = data['Hold Times'].apply(literal_eval)
    data['Flight Times'] = data['Flight Times'].apply(literal_eval)
    data['Press/Release Timings'] = data['Press/Release Timings'].apply(literal_eval)
    data['Key Combinations'] = data['Key Combinations'].apply(literal_eval)

    # Find the maximum length among the lists in each column
    max_len_hold_times = max(data['Hold Times'].apply(len))
    max_len_flight_times = max(data['Flight Times'].apply(len))
    max_len_press_release = max(data['Press/Release Timings'].apply(len))
    max_len_key_combinations = max(data['Key Combinations'].apply(len))

    # Pad the lists with zeros to make them of equal length
    data['Hold Times'] = data['Hold Times'].apply(lambda x: x + [0] * (max_len_hold_times - len(x)))
    data['Flight Times'] = data['Flight Times'].apply(lambda x: x + [0] * (max_len_flight_times - len(x)))
    data['Press/Release Timings'] = data['Press/Release Timings'].apply(lambda x: x + [(0, 0)] * (max_len_press_release - len(x)))
    data['Key Combinations'] = data['Key Combinations'].apply(lambda x: x + [('0', '0')] * (max_len_key_combinations - len(x)))

    # Convert the lists to arrays
    data['Hold Times'] = data['Hold Times'].apply(np.array)
    data['Flight Times'] = data['Flight Times'].apply(np.array)
    data['Press/Release Timings'] = data['Press/Release Timings'].apply(np.array)
    data['Key Combinations'] = data['Key Combinations'].apply(np.array)

    # Normalize the Hold Times and Flight Times using MinMaxScaler
    scaler = MinMaxScaler()
    data['Hold Times'] = data['Hold Times'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    data['Flight Times'] = data['Flight Times'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())

    # Return preprocessed features
    return data

# Apply preprocessing to training and testing features
X_train_preprocessed = preprocess_features(X_train)
X_test_preprocessed = preprocess_features(X_test)

# Verify the shape of the Hold Times column
print(X_train_preprocessed['Hold Times'].shape)
print(X_test_preprocessed['Hold Times'].shape)

# Reshape the input shape for the MLP model
input_shape = (X_train_preprocessed['Hold Times'][0].shape[0],)
