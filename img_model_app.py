import numpy as np
import cv2
import os
import streamlit as st
import pickle
import time


st.set_page_config(page_title="Image Classification",
                   page_icon=":eye:", layout="wide")

with open('logistic_classifier.pkl', 'rb') as model:
    loaded_model = pickle.load(model)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

buildir = 'Image Classification/building'
roadir = 'Image Classification/road'

with st.container():
    st.title("Welcome to the Image Classification App :wave:")
    left_column, right_column = st.columns((1, 2))
    with left_column:
        img = st.file_uploader(
            "Please choose an image to Classify! :heart_eyes:")
    with right_column:
        if img != None:
            st.image(img)
            st.write(img.name)

with st.container():
    st.write("---")
    st.write("##")
    st.markdown("<h2 style='text-align: center;'>Result</h2>",
                unsafe_allow_html=True)
    if img != None:
        for a in os.listdir(buildir):
            if img.name == a:
                path = os.path.join(buildir, img.name)
        for a in os.listdir(roadir):
            if img.name == a:
                path = os.path.join(roadir, img.name)
        img = cv2.imread(path, 0)
        resize = cv2.resize(img, (150, 150))
        flat_img = resize.flatten()
        flat_img = np.array(flat_img)
        prediction = loaded_model.predict(flat_img.reshape(1, -1))
        if prediction[0] == 0:
            progress_bar = st.progress(0)
            for perc_completed in range(100):
                time.sleep(0.05)
                progress_bar.progress(perc_completed+1)
            st.markdown(
                "<h5 style='text-align: center;'>It's <b>BUILDING</b></h5>", unsafe_allow_html=True)
        else:
            progress_bar = st.progress(0)
            for perc_completed in range(100):
                time.sleep(0.05)
                progress_bar.progress(perc_completed+1)
            st.markdown(
                "<h5 style='text-align: center;'>It's <b>ROAD</b></h5>", unsafe_allow_html=True)

    st.write("---")
    st.markdown("<h3 style='text-align: center;'>Thank You for Visiting!</h3>",
                unsafe_allow_html=True)
