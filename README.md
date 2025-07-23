# ksense: Real-time Emotion Detection & Mental Wellness Alert System

## Features
- Real-time webcam emotion detection (DeepFace + OpenCV)
- Persistent alert triggering for negative emotions
- Auto email notification
- Minimal UI with Streamlit
- Emotion trend graph
- Data privacy controls via .env
- Modular, production-style code structure

## Setup
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Fill in `.env` with your email credentials
4. Run: `streamlit run app/main.py`

## Deployment
- Ready for Streamlit Cloud or Hugging Face Spaces

## Folder Structure
```
ksense/
├── app/
│   ├── main.py           # Streamlit app
│   ├── detect.py         # DeepFace wrapper
│   ├── alert.py          # Alert logic
│   ├── camera.py         # Webcam capture
│   ├── plot.py           # Emotion trend graphs
│   └── utils.py          # Helpers
├── .env
├── requirements.txt
├── README.md
``` 