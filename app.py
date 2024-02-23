import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Classifying Iris Flowers using ML')
st.markdown('Toy model to play to classify iris flowers into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])

![1](https://github.com/Rahul-Biju-03/Rahul-Biju-03/assets/106422354/1720941e-a1c3-4efe-b776-338489815c9b)

st.text('')
st.text('')
st.markdown(
    '`Create by` [Rahul Biju](https://github.com/Rahul-Biju-03/Deployment-repo/blob/main/README.md) | \
         `Code:` [GitHub](https://github.com/Rahul-Biju-03/Deployment-repo/blob/main/app.py)')
