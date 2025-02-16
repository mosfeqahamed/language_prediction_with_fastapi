import pickle
import re
from pathlib import Path

# Define the base directory for the model
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the model dynamically without specifying a version
try:
    model_path = next(BASE_DIR.glob("trained_pipeline-*.pkl"))  # Find any version of the model file
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except StopIteration:
    raise RuntimeError("No trained pipeline file found in the directory.")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Define the classes for language prediction
classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

# Function to preprocess input text
def preprocess_text(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    return text.lower()

# Function to predict the language of a text
def predict_pipeline(text):
    text = preprocess_text(text)
    try:
        pred = model.predict([text])  # This works regardless of scikit-learn version
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
    return classes[pred[0]]
