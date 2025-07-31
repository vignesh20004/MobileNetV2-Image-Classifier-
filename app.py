# enhanced_image_classifier.py
import os
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 15

# Data preparation with augmentation
def prepare_data(train_dir, val_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, val_generator

# Model builder with fine-tuning
def build_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training with callbacks
def train_model(model, train_generator, val_generator):
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=2, verbose=1)
    ]
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return model

# Evaluation
def evaluate_model(model, val_generator, class_names):
    val_generator.reset()
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Prediction
def predict_and_visualize(model, image_input, class_names, true_class=None):
    try:
        img = Image.open(image_input).convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        st.image(img, caption=f"Prediction: {predicted_class} ({confidence*100:.2f}%)", use_column_width=True)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main
if __name__ == "__main__":
    st.title("Image Classifier with MobileNetV2")

    train_dir = r'\dataset\train'
    val_dir = r'\dataset\test'

    train_generator, val_generator = prepare_data(train_dir, val_dir)
    class_names = list(train_generator.class_indices.keys())

    model = build_model(NUM_CLASSES)
    model = train_model(model, train_generator, val_generator)

    evaluate_model(model, val_generator, class_names)
    model.save('object_classifier.h5')

    uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        predict_and_visualize(model, uploaded_file, class_names)