import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import callbacks
import pickle

# Load the dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
data.head()

# Split the data into features and target
X = data.drop(["DEATH_EVENT"], axis=1)
Y = data["DEATH_EVENT"]

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Save the scaler for later use in the Streamlit app
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.25, random_state=7)

# Define early stopping callback
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

# Initialize the neural network
model = Sequential()

# Add layers
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, batch_size=32, epochs=500, callbacks=[early_stopping], validation_split=0.2)

# Calculate and print validation accuracy
val_accuracy = np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy * 100))

# Save the trained model
filename = 'sakit_jantung.sav'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model has been saved as {filename}")

# Example prediction with the trained model
input_data = (75, 0, 582, 0, 20, 1, 265000, 1.9, 130, 1, 0, 4)
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshape)
print(std_data)

# Make prediction
prediction = model.predict(std_data)
print(prediction)

if prediction[0][0] > 0.5:
    print('The patient has died.')
else:
    print('The patient is alive.')

