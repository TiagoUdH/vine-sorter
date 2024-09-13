import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

import gdown
import io

@st.cache_resource
def loading_model():
    url = "https://drive.google.com/uc?id=1l6WH_-5a3715Z7j4IsxwxIrXwmOKjpVh"

    gdown.download(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def loading_image():
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success("Imagem carregada com sucesso")

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis = 0)

        return image

def prevision(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['LeafBlight', 'HealthyGrapes', 'BlackRot', 'BlackMeasles']

    df = pd.DataFrame()
    df["classes"] = classes
    df["probabilidades (%)"] = 100 * output_data[0]

    fig = px.bar(df, y="classes", x="probabilidades (%)", orientation='h', text="probabilidades (%)", title="Probabilidade das classes de Doenças de Uvas")

    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classificador de folhas de videira",
        page_icon="🍇"
    )

    st.write("# Classificador de folhas de videira 🍇")

    interpreter = loading_model()
    image = loading_image()

    if image is not None:
        prevision(interpreter, image)

if __name__ == "__main__":
    main()