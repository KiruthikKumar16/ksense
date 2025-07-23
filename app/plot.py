import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def plot_emotion_trend(df, return_fig=False):
    if df.empty:
        return None if return_fig else None
    plt.figure(figsize=(8, 3))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    plt.plot(df['timestamp'], df['emotion'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.title('Emotion Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if return_fig:
        fig = plt.gcf()
        plt.close(fig)
        return fig
    else:
        st.pyplot(plt) 