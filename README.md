# Rock Paper Scissors Gesture Recognition

This project is a small prototype for hand gesture recognition using MediaPipe and TensorFlow.

The goal is to recognize three hand gestures:

- Rock
- Paper
- Scissors

The system uses MediaPipe to extract 21 hand landmarks from images. Each landmark contains x, y, and z coordinates, so each image is converted into 63 numerical features. A TensorFlow neural network model is then trained to classify the gesture.

## Project Structure

```text
gesture/
├── paper/
├── rock/
├── scissors/
├── create_custom_dataset.py
├── train_model.py
├── predict_image.py
├── gesture_landmarks_custom.csv
├── gesture_tf_model.keras
├── label_encoder.pkl
├── hand_landmarker.task
├── training_accuracy.png
├── training_loss.png
├── confusion_matrix.png
├── requirements.txt
└── README.md
```
## Workflow
Collect hand gesture images
↓
Extract hand landmarks using MediaPipe
↓
Save landmarks as CSV
↓
Train TensorFlow model
↓
Predict new gesture image


## Dataset
paper scissors rock each class currently contains around 30 images.

## Model
The model is a TensorFlow Dense Neural Network.

Input:63 hand landmark features
Output:rock / paper / scissors

# How to Run

1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies

python -m pip install -r requirements.txt

3. Create custom landmark dataset
python create_custom_dataset.py

This will generate:gesture_landmarks_custom.csv

4. Train model
python train_model.py

This will generate:
- gesture_tf_model.keras
- label_encoder.pkl
- training_accuracy.png
- training_loss.png
- confusion_matrix.png

5. Predict a new image

Put a test image in the project folder and update the image path in predict_image.py.

python predict_image.py
