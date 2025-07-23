import cv2
from deepface import DeepFace
from fer import FER
import numpy as np

# Initialize FER detector once
fer_detector = FER(mtcnn=True)

# Returns (dominant_emotion, emotions_dict) for both models

def analyze_emotion(frame, model='deepface'):
    if model == 'fer':
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = fer_detector.detect_emotions(rgb_frame)
        if result and 'emotions' in result[0]:
            emotions = result[0]['emotions']
            dominant = max(emotions, key=emotions.get)
            return dominant, emotions
        return 'unknown', {}
    else:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            dominant = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
            return dominant, emotions
        except Exception as e:
            return "unknown", {} 