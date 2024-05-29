
from tensorflow.keras.models import load_model

# Load the model
best_model_path = 'ResNet50_Transfer_Learning.h5'
try:
    model = load_model(best_model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

