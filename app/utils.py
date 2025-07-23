import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import os

def log_emotion(emotion):
    if 'emotion_log' not in st.session_state:
        st.session_state['emotion_log'] = []
    st.session_state['emotion_log'].append((datetime.now(), emotion))

def load_env():
    load_dotenv()

def get_env(key):
    return os.getenv(key) 