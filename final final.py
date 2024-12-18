import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
file_path = 'C:/Users/Tamish Sridatta.B/OneDrive/Desktop/College/Intership/ClaMP_Integrated-5184.csv'
df = pd.read_csv(file_path)

# Replace 'actual_target_column_name' with the actual name of your target column
target_column_name = 'class'  # Update to the correct column name

# Separate features and target
X = df.drop(columns=[target_column_name])
y = df[target_column_name]

# Check for NaN values
print(f"NaN values in dataset: {df.isnull().sum().sum()}")

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_columns}")

# Preprocess the data
# OneHotEncode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', StandardScaler(), X.select_dtypes(include=['number']).columns)
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Verify shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Model definition
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)  # Adjust learning rate if necessary
model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.tight_layout()
plt.show()
