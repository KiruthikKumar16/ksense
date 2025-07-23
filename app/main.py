import streamlit as st
import cv2
import time
import numpy as np
from detect import analyze_emotion
from camera import get_webcam_frame
from plot import plot_emotion_trend
from alert import check_and_alert
from utils import log_emotion, log_emotion_csv, get_csv_log, load_env
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="ksense: Real-time Emotion Detection", layout="centered")
st.title("ksense: Real-time Emotion Detection & Mental Wellness Alert System")

# Load environment variables
load_env()

# --- Model selection ---
model_choice = st.selectbox("Select Emotion Detection Model", ["deepface", "fer"], index=1)

# --- Camera device selection ---
camera_index = st.number_input("Camera Index (0=default)", min_value=0, max_value=10, value=0, step=1)

# --- Live detection controls ---
live_placeholder = st.empty()
start = live_placeholder.button("Start Live Detection")
stop = st.button("Stop Live Detection")

# --- Session state for live detection ---
if 'live_running' not in st.session_state:
    st.session_state['live_running'] = False
if start:
    st.session_state['live_running'] = True
if stop:
    st.session_state['live_running'] = False

# --- Emotion log state ---
def get_emotion_log():
    if 'emotion_log' not in st.session_state:
        st.session_state['emotion_log'] = []
    return st.session_state['emotion_log']

# --- CSV log download button ---
if os.path.isfile('emotion_log.csv'):
    with open('emotion_log.csv', 'rb') as f:
        st.download_button('Download Emotion Log (CSV)', f, file_name='emotion_log.csv', mime='text/csv')

# --- Live webcam detection ---
FRAME_WINDOW = st.empty()
emotion_text = st.empty()
trend_placeholder = st.empty()
timeline_placeholder = st.empty()

N_HISTORY = 10  # Number of last emotions to show (increased from 3 to 10)
FRAME_DELAY = 0.3  # seconds between frames

# --- Supported emotions legend ---
supported_emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
st.markdown(
    "**Supported Emotions:** " + ", ".join(supported_emotions)
)

bar_placeholder = st.empty()

if st.session_state['live_running']:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Webcam (index {camera_index}) not accessible.")
        st.session_state['live_running'] = False
    else:
        while st.session_state['live_running']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam.")
                st.session_state['live_running'] = False
                break
            frame = cv2.flip(frame, 1)
            face_results = analyze_emotion(frame, model=model_choice)
            # Draw bounding boxes and labels
            display_frame = frame.copy()
            face_labels = []
            for face in face_results:
                box = face['box']
                dominant_emotion = face['dominant_emotion']
                emotions_dict = face['emotions']
                confidence = emotions_dict.get(dominant_emotion, None)
                if box is not None:
                    x, y, w, h = box if len(box) == 4 else (box['x'], box['y'], box['w'], box['h'])
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{dominant_emotion}"
                    if confidence is not None:
                        label += f" ({confidence:.2%})"
                    cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    face_labels.append(label)
            FRAME_WINDOW.image(display_frame, channels="BGR")
            # List all detected faces/emotions for this frame
            if face_labels:
                emotion_text.markdown("**Detected Faces:**<br>" + "<br>".join(face_labels), unsafe_allow_html=True)
            else:
                emotion_text.markdown("**No face detected**")
            # Log the first face's emotion for session/alert/logging
            if face_results:
                log_emotion(face_results[0]['dominant_emotion'])
                log_emotion_csv(
                    timestamp=pd.Timestamp.now().isoformat(),
                    emotion=face_results[0]['dominant_emotion'],
                    model=model_choice,
                    confidence=face_results[0]['emotions'].get(face_results[0]['dominant_emotion'], '')
                )
                check_and_alert(get_emotion_log())
            # Show trend graph
            df = pd.DataFrame(get_emotion_log(), columns=["timestamp", "emotion"])
            if not df.empty:
                trend_placeholder.pyplot(plot_emotion_trend(df, return_fig=True))
            # Show last N emotions
            last_n = get_emotion_log()[-N_HISTORY:]
            timeline_placeholder.markdown(
                "**Last {} Emotions:** ".format(N_HISTORY) +
                ", ".join([f"{e[1]}" for e in last_n])
            )
            # Show bar chart for first face
            if face_results and face_results[0]['emotions']:
                emotions_dict = face_results[0]['emotions']
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.bar(emotions_dict.keys(), emotions_dict.values(), color='skyblue')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Emotion Probabilities (First Face)')
                bar_placeholder.pyplot(fig)
            else:
                bar_placeholder.empty()
            time.sleep(FRAME_DELAY)
        cap.release()

# --- Image upload option ---
uploaded = st.file_uploader("Or upload an image for analysis", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    face_results = analyze_emotion(img, model=model_choice)
    display_img = img.copy()
    face_labels = []
    for face in face_results:
        box = face['box']
        dominant_emotion = face['dominant_emotion']
        emotions_dict = face['emotions']
        confidence = emotions_dict.get(dominant_emotion, None)
        if box is not None:
            x, y, w, h = box if len(box) == 4 else (box['x'], box['y'], box['w'], box['h'])
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{dominant_emotion}"
            if confidence is not None:
                label += f" ({confidence:.2%})"
            cv2.putText(display_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            face_labels.append(label)
    st.image(display_img, channels="BGR", caption=f"Detected Faces: {', '.join(face_labels) if face_labels else 'None'}")
    if face_results and face_results[0]['emotions']:
        emotions_dict = face_results[0]['emotions']
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(emotions_dict.keys(), emotions_dict.values(), color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Probabilities (First Face)')
        st.pyplot(fig)

# --- Session summary section ---
st.markdown('---')
st.header('Session Summary')
log_data = get_emotion_log()
if log_data:
    df_summary = pd.DataFrame(log_data, columns=["timestamp", "emotion"])
    emotion_counts = df_summary['emotion'].value_counts()
    emotion_percent = emotion_counts / emotion_counts.sum() * 100
    st.subheader('Emotion Distribution')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    st.subheader('Emotion Counts')
    st.dataframe(df_summary['emotion'].value_counts().reset_index().rename(columns={'index': 'Emotion', 'emotion': 'Count'}))
    # Show session duration
    start_time = df_summary['timestamp'].iloc[0]
    end_time = df_summary['timestamp'].iloc[-1]
    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    st.info(f"Session Duration: {duration}")
else:
    st.info('No session data yet. Start live detection to see your session summary.') 