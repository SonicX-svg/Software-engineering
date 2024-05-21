commit 4a5e34e44bf424f7c944837a22e4ba4943f563d5
Author: SonicX-svg <143742185+SonicX-svg@users.noreply.github.com>
Date:   Tue May 21 17:50:24 2024 +0300

    Update README.md
    
    add more detailed description

diff --git a/README.md b/README.md
index 1881991..930b0d0 100644
--- a/README.md
+++ b/README.md
@@ -1,4 +1,9 @@
 # Software-engineering
 ## <ins> Код улучшен с использованием форматера black и общим структурированием проекта. </ins> <br>
+1. Оформлен по стандарту PEP8 (black)
+2. Имеет осмысленные переменные
+3. Код функционально структурирован
+4. Проект также имеет структуру
+-----
 Web-приложение предсказания рака груди на основе набора данных Breast_cancer и модели knn. <br>
 Приложение доступно по ссылке: https://software-engineering-peffhxxvxl4u4euk59q5hj.streamlit.app/
# importing important libraries
import streamlit as st
import numpy as np
import time
import pickle
import sklearn
import pandas as pd
import random
import os
import sys
from pathlib import Path

# Расчёт пути к родительской директории
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Проверка наличия родительской директории в sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

data = pd.read_csv("data/breast_cancer.csv")
# Open the PKL file
with open("model/neigh_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

# column name for each feature in the diabetes dataset.
column_names = data.columns
print(column_names, len(column_names))


# function to receive users' information.
def inputs():
    # creating the form for data inputs.
    with st.form(key="diabetes_data"):
        name = st.text_input("Patient's Name: ")
        gender_obj = st.selectbox(label="Patient's Gender: ", options=["Female"])
        if gender_obj == "Female":
            gender = 1
        worst_concave_points = st.number_input(label="worst concave points: ")
        worst_perimeter = st.number_input(label="worst perimeter: ")
        mean_concave_points = st.number_input(label="mean concave points: ")
        worst_radius = st.number_input(label="worst_radius: ")
        mean_perimeter = st.number_input(label="mean compactness: ")
        worst_area = st.number_input(label="mean_perimeter: ")
        mean_radius = st.number_input(label="mean_radius: ")
        mean_area = st.number_input(label="mean_area: ")
        mean_concavity = st.number_input(label="mean_concavity: ")
        worst_concavity = st.number_input(label="worst_concavity: ")
        submit = st.form_submit_button("Submit Test")
        if submit:
            patient_data = [
                worst_concave_points,
                worst_perimeter,
                mean_concave_points,
                worst_radius,
                mean_perimeter,
                worst_area,
                mean_radius,
                mean_area,
                mean_concavity,
                worst_concavity,
            ]
        else:
            patient_data = [0 for i in range(10)]
    return patient_data


# function to create a data frame and carry out prediction.
def predict(var_name):
    pred = [var_name]
    np_pred = np.array(pred)
    score = knn_model.predict(np_pred)
    return score


# function to run streamlit app
def run():
    st.title("Cancer breast Test App")

    info = inputs()
    dia_score = predict(info)
    with st.spinner(text="Diagnosing....."):
        time.sleep(5)
    if dia_score == 1:
        st.error("Positive. Cancer Diagnosed.")
    else:
        st.success("Negative. Cancer not diagnosed.")


# running streamlit app.
if __name__ == "__main__":
    run()
