import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define image dimensions
img_width, img_height = 224, 224

# Load the dataset
data = pd.read_csv('/media/guimcc/Elements/data_2/train.csv')

# Extract image paths and numerical values
image_paths = data['image_path'].tolist()
numerical_values = data[[str(i) for i in range(12)]].astype('float32')

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    try:
        img = load_img(f'/media/guimcc/Elements/data_2/images/{image_path}', target_size=(img_width, img_height))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        # If there's an error loading the image, return an array of zeros
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((img_height, img_width, 3))

# Load and preprocess images
images = np.array([load_and_preprocess_image(path) for path in image_paths])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, numerical_values, test_size=0.2, random_state=42
)

# Define the CNN model
input_shape = (img_height, img_width, 3)
inputs = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(12, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
