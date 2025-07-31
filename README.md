# 🖼️ MobileNetV2 Image Classifier (Streamlit App)

This project is an advanced image classification web application built with **Streamlit** and **MobileNetV2**, trained on a custom image dataset. It allows users to upload images and get real-time predictions with confidence scores.

## 🚀 Features

- Transfer learning using **MobileNetV2**
- Real-time prediction via **Streamlit**
- Live webcam/file image uploader support
- Evaluation metrics with confusion matrix and classification report
- Uses data augmentation and callbacks for robust training

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pillow
- scikit-learn
- seaborn
- Matplotlib

## 📁 Directory Structure
├── app.py # Main Streamlit app
├── dataset/ # Training and validation image folders
│ ├── train/
│ └── test/
├── object_classifier.h5 # Trained model file (auto-generated after training)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore rules

## Structure your dataset like this:

     
     dataset/
       train/
         animals/ (cat, tiger, etc. - 10 images each)
         birds/
         vehicles/
         etc/
       Validation/
         animals/
         birds/
         vehicles/
         etc/
## Optional:
After training, object_classifier.h5 is saved automatically.

Modify NUM_CLASSES and paths as per your dataset.


## ✨ Author
Vighnesh M. 



