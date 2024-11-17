import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Setup Paths
BASE_DIR = r'C:\Users\junos\OneDrive\Documents\PythonScripts\dataset'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Target image size for resizing
IMG_SIZE = (224, 224)

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =============================
# 1. Data Preparation
# =============================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # Reserve 20% for validation
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

# Training and validation data
train_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Training'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Training'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Test'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)

# =============================
# 2. Build the Model
# =============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =============================
# 3. Train the Model
# =============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Reduce for testing; use more epochs (e.g., 25) for full training
)

# =============================
# 4. Evaluate and Save Model
# =============================
# Evaluate on test data
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Save the model in Keras format
model.save('binary_classification_model.keras')

# =============================
# 5. Visualize Training History
# =============================
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =============================
# 6. Classification Report
# =============================
# Get predictions on the test data
test_data.reset()
y_pred = np.round(model.predict(test_data))
y_true = test_data.classes

# Print classification report
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

# =============================
# 7. Flask Deployment
# =============================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        label = "Class 1" if prediction > 0.5 else "Class 2"
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

        return render_template(
            'index.html',
            prediction=label,
            confidence=confidence,
            image_url=file_path
        )


if __name__ == '__main__':
    app.run(debug=True)
