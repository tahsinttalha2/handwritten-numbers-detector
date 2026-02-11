import sys
import os

# add the current folder to python's search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import torch
import torchvision.transforms as transforms
from main import HandNumDetector
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = HandNumDetector()
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("HNmodel1.pth", map_location= torch.device(gpu_device)))
    model.eval()

    return model

model = load_model()

st.title("Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit and see if the model can read that!")

model_name = st.selectbox(
    "Choose your desired Model",
    ("HNmodel v1.0"),
)

image = st.file_uploader(
    label = "Upload an Image of a Number",
    type = ["jpg", "jpeg", "png"]
)

if image is not None:
    image = Image.open(image)

    st.image(image, caption="User Image", width = 100)

    #convert image to grayscale
    image = image.convert("L")

    if np.array(image) [0, 0] > 128:
        image = ImageOps.invert(image)  
        st.caption("Auto inverted colour (detected white background)")  

    #resize the image
    image = image.resize((28, 28))

    #display converted image
    st.image(image.resize((100, 100), resample = Image.NEAREST), caption = "Model View (28x28)", width = 100)

    #convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image).unsqueeze(0)

    if st.button("Make Prediction"):
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim = 0)

            confidence, predicted = torch.max(probabilities, 0)

            st.success(f"Prediction: {predicted.item()}")
            st.info(f"Confidence: {confidence.item() * 100:.2f}%")

            prob_dict = {f"{i}": float(probabilities[i]) for i in range(10)}
            st.bar_chart(prob_dict)

