import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import os
import csv

def log_emotion(emotion):
    if 'emotion_log' not in st.session_state:
        st.session_state['emotion_log'] = []
    st.session_state['emotion_log'].append((datetime.now(), emotion))

def log_emotion_csv(timestamp, emotion, model, confidence, csv_path='emotion_log.csv'):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'emotion', 'model', 'confidence'])
        writer.writerow([timestamp, emotion, model, confidence])

def get_csv_log(csv_path='emotion_log.csv'):
    if not os.path.isfile(csv_path):
        return ''
    with open(csv_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_env():
    load_dotenv()

def get_env(key):
    return os.getenv(key) 