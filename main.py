import streamlit as st
import pandas as pd
import numpy as np 
import soundfile as sf 

st.header(' For Visually Imapired People') 
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(f'filename: {uploaded_file.name}')
    st.image(bytes_data)
    if st.button('Translate'):
        audio_path = "/home/fm-pc-lt-269/Documents/Fusemachines/Practice/VQA/bark_out.wav"
        audio_bytes,_ = sf.read(audio_path)
        st.audio(audio_bytes, format='audio/wav',sample_rate=16000)
    else:
        st.write('Goodbye')


    # prompt = st.chat_input("Say something")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")

