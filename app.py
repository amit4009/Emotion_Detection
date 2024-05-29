
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load the pre-trained model
best_model_path = 'ResNet50_Transfer_Learning.h5'
model = load_model(best_model_path)
print("Model loaded successfully!")

# Define emotion classes
Emotion_Classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(image_path):
    # Load image in color mode (to get 3 channels)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Resize the image to the size expected by the model
    img = cv2.resize(img, (224, 224))
    # Normalize the pixel values
    img = img.astype('float32') / 255.0
    # Reshape to add batch dimension (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Make prediction
    predictions = model.predict(img)
    # Get the predicted class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]
    # Get the probability of the predicted class
    predicted_probability = np.max(predictions)
    return Emotion_Classes[predicted_class], float(predicted_probability)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img_path = 'temp.jpg'
        file.save(img_path)
        
        # Ensure the file is closed properly
        file.stream.close()
        
        emotion, probability = predict_emotion(img_path)
        
        # Remove the file after processing
        try:
            os.remove(img_path)  # Clean up the temporary file
        except Exception as e:
            print(f"Error removing file: {e}")
        
        return jsonify({'emotion': emotion, 'probability': probability})

if __name__ == "__main__":
    app.run(debug=True)

