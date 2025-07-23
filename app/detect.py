import cv2
from deepface import DeepFace
from fer import FER
import numpy as np

# Initialize FER detector once
fer_detector = FER(mtcnn=True)

# Returns a list of dicts: [{box, dominant_emotion, emotions_dict}]
def analyze_emotion(frame, model='deepface'):
    results = []
    if model == 'fer':
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = fer_detector.detect_emotions(rgb_frame)
        for face in faces:
            emotions = face['emotions']
            dominant = max(emotions, key=emotions.get)
            box = face['box']  # (x, y, w, h)
            results.append({'box': box, 'dominant_emotion': dominant, 'emotions': emotions})
        return results
    else:
        try:
            # DeepFace does not natively support multi-face, so fallback to single face
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                for r in result:
                    emotions = r['emotion']
                    dominant = r['dominant_emotion']
                    box = r.get('region', None)  # DeepFace may return region as box
                    results.append({'box': box, 'dominant_emotion': dominant, 'emotions': emotions})
            else:
                emotions = result['emotion']
                dominant = result['dominant_emotion']
                box = result.get('region', None)
                results.append({'box': box, 'dominant_emotion': dominant, 'emotions': emotions})
            return results
        except Exception as e:
            return [] 