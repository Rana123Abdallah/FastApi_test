from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow 
import librosa
from io import BytesIO

model = tensorflow.keras.models.load_model(r'D:\Api_model-main\Api_model-main\model.h4')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioModel(BaseModel):
    file: UploadFile

for layer in model.layers:    
    layer.trainable = False    
'''
def process_audio(file_bytes):

    try:
        y, sr = librosa.load(file_bytes, sr=22050)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = np.expand_dims(mfcc, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.expand_dims(chroma, axis=0)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.expand_dims(spectrogram, axis=0)

    try:
        mfcc_output = model([mfcc, chroma, spectrogram])
    except Exception as e:
        raise ValueError(f"Failed to make prediction: {e}")

    mylist = {
        '0': 'URTI',
        '1': 'Healthy',
        '2': 'Asthma',
        '3': 'COPD',
        '4': 'LRTI',
        '5': 'Bronchiectasis',
        '6': 'Pneumonia',
        '7': 'Bronchiolitis'
    }

    prediction = np.argmax(mfcc_output)

    for key, predicted_label in mylist.items():
        if key == str(prediction):
            return predicted_label

    return predicted_label

@app.post("/predict")
def predict_audio(file: UploadFile = File(...)):
    # Read the file contents into a BytesIO object
    file_bytes = BytesIO(file.file.read())
    # Call the process_audio function
    result = process_audio(file_bytes)
    return {"result": result}
'''
def process_audio(file_bytes):

    try:
        y, sr = librosa.load(file_bytes, sr=22050)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")
    # Define the window size and stride
    window_size = 6 * sr # 6 seconds
    stride = 1 * sr # 1 second

    # Initialize an empty list to store the predictions
    predictions = []

    # Iterate over the full voice recording with a sliding window
    for i in range(0, len(y) - window_size, stride):
        # Extract the current window
        window = y[i:i+window_size]
    
    # Preprocess the window
    window = librosa.util.fix_length(window,size=window_size)

    mfcc = librosa.feature.mfcc(y=window, sr=sr)
    mfcc = np.expand_dims(mfcc, axis=0)

    chroma = librosa.feature.chroma_stft(y=window, sr=sr)
    chroma = np.expand_dims(chroma, axis=0)

    spectrogram = librosa.feature.melspectrogram(y=window, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.expand_dims(spectrogram, axis=0)

    try:
        mfcc_output = model([mfcc, chroma, spectrogram])
    except Exception as e:
        raise ValueError(f"Failed to make prediction: {e}")

    mylist = {
        '0': 'URTI',
        '1': 'Healthy',
        '2': 'Asthma',
        '3': 'COPD',
        '4': 'LRTI',
        '5': 'Bronchiectasis',
        '6': 'Pneumonia',
        '7': 'Bronchiolitis'
    }

    prediction = np.argmax(mfcc_output)

    for key, predicted_label in mylist.items():
        if key == str(prediction):
            return predicted_label

    return predicted_label

@app.post("/predict")
def predict_audio(file: UploadFile = File(...)):
    # Read the file contents into a BytesIO object
    file_bytes = BytesIO(file.file.read())
    # Call the process_audio function
    result = process_audio(file_bytes)
    return {"result": result}