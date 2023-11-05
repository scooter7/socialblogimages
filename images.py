import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import keras_ocr
import cv2
import numpy as np
import math

nltk.download('punkt')
nltk.download('stopwords')

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if not word.lower() in stop_words]
    return ' '.join(filtered_text)

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(image, pipeline):
    prediction_groups = pipeline.recognize([image])
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
    inpainted_img = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    return inpainted_img

def generate_image(prompt, width, height):
    prompt += " | only picture, no text"
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=f"{width}x{height}"
    )
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image

pipeline = keras_ocr.pipeline.Pipeline()

st.title('Text to Image Generator')

width = st.number_input('Width', min_value=256, max_value=1024, value=512)
height = st.number_input('Height', min_value=256, max_value=1024, value=512)

user_input = st.text_area("Enter your text here:")

if st.button('Generate Image'):
    keywords = extract_keywords(user_input)
    image = generate_image(keywords, width, height)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_text_removed = inpaint_text(image_np, pipeline)
    st.image(img_text_removed, caption='Generated Image', channels="BGR")
