import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import json
from streamlit_lottie import st_lottie

# resolve error NotImplementedError: cannot instantiate 'WindowsPath' on your system
import pathlib
plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

#--------------------------#
# customise app UI

st.set_page_config(
    page_title="Bee Classifier",
    page_icon="🐝",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a personal project to identify bee species."
    }
)


#--------------------------#

# lottie file (in images)


st.markdown("<h1 style='text-align: center;'>Welcome, I am your personal Bee species classifier.</h1>",
            unsafe_allow_html=True)


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_coding = load_lottiefile("images/bee-flying.json")
col1, col2, col3 = st.columns(3)
with col2:
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium",  # medium ; high,
        key=None,
    )


#--------------#
def load_model():
    return load_learner("bee_init.pkl")


with st.spinner("I am collecting my thoughts..."):
    model = load_model()

# create a file uploader in Streamlit and display an uploaded image
uploaded_image = st.file_uploader(
    "Upload your image and I'll give it a try.", type=["png", "jpg"])

#--------------#
if uploaded_image is not None:

    st.image(uploaded_image)

    try:
        pred, pred_idx, probs = model.predict(uploaded_image.getvalue())
        st.success(
            # f"{bee_dict[pred]} I am {probs[pred_idx]*100:.0f}% confident.")
            f"{[pred]} I am {probs[pred_idx]*100:.0f}% confident.")
        st.caption(
            f"Caution: I have only been trained on a small set of images. I may also be wrong.")
    except:
        st.write("Sorry, I don't know that fruit")

#--------------#
