import streamlit as st
import cv2
import time
import numpy as np
from detect import analyze_emotion
from camera import get_webcam_frame
from plot import plot_emotion_trend
from alert import check_and_alert
from utils import log_emotion, load_env
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
            detected_emotion, emotions_dict = analyze_emotion(frame, model=model_choice)
            FRAME_WINDOW.image(frame, channels="BGR")
            emotion_text.markdown(f"**Detected Emotion:** {detected_emotion}")
            log_emotion(detected_emotion)
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
            # Show bar chart of all emotion probabilities
            if emotions_dict:
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.bar(emotions_dict.keys(), emotions_dict.values(), color='skyblue')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Emotion Probabilities')
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
    detected_emotion, emotions_dict = analyze_emotion(img, model=model_choice)
    st.image(img, channels="BGR", caption=f"Detected Emotion: {detected_emotion}")
    if emotions_dict:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(emotions_dict.keys(), emotions_dict.values(), color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Probabilities')
        st.pyplot(fig) 