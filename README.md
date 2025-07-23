# ksense: Real-time Emotion Detection & Mental Wellness Alert System



## Features
- Real-time webcam emotion detection (DeepFace + FER + OpenCV)
- Multiple face detection and bounding boxes
- Persistent alert triggering for negative emotions
- Auto email notification (customizable)
- Minimal UI with Streamlit
- Emotion trend graph and bar chart
- Data privacy controls via .env
- Modular, production-style code structure
- CSV logging of all detected emotions
- Session summary (pie chart, table, duration)
- Model selection (DeepFace/FER)
- Camera device selection
- Confidence score display
- Downloadable emotion log (CSV)
- Start/Stop live detection
- Robust error handling

## How it works
- Select your model and camera, then start live detection.
- The app detects all faces, draws bounding boxes, and shows their emotions and confidence.
- A real-time trend graph, bar chart, and timeline are shown.
- All data is logged to CSV and can be downloaded.
- At the end, see a session summary with emotion distribution and duration.

## Setup
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Fill in `.env` with your email credentials (for alerts)
4. Run: `streamlit run app/main.py`

## Deployment
- Ready for Streamlit Cloud or Hugging Face Spaces
- Click the badge above to deploy instantly

## Folder Structure
```
ksense/
├── app/
│   ├── main.py           # Streamlit app
│   ├── detect.py         # DeepFace/FER wrapper
│   ├── alert.py          # Alert logic
│   ├── camera.py         # Webcam capture
│   ├── plot.py           # Emotion trend graphs
│   └── utils.py          # Helpers
├── .env
├── requirements.txt
├── README.md
├── .gitignore
```

## Privacy
- No images or video are stored unless you enable logging.
- All logs are local and downloadable by the user only.

## .gitignore
```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Environment
.venv/
venv/
.env

# Logs and data
emotion_log.csv

# OS files
.DS_Store
Thumbs.db
``` 
