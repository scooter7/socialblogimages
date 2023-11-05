import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import easyocr
import cv2
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if not word.lower() in stop_words]
    return ' '.join(filtered_text)

def remove_text(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        if prob >= 0.6:
            top_left = tuple(bbox[0][0])
            bottom_right = tuple(bbox[2][0])
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    return image

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

st.title('Text to Image Generator')

width = st.number_input('Width', min_value=256, max_value=1024, value=512)
height = st.number_input('Height', min_value=256, max_value=1024, value=512)

user_input = st.text_area("Enter your text here:")

if st.button('Generate Image'):
    keywords = extract_keywords(user_input)
    image = generate_image(keywords, width, height)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_no_text = remove_text(image_np)
    st.image(cv2.cvtColor(image_no_text, cv2.COLOR_BGR2RGB), caption='Generated Image')

