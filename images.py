import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pytesseract
import cv2
import numpy as np

# Make sure to download these once using nltk.download if they are not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if not word.lower() in stop_words]
    return ' '.join(filtered_text)

def remove_text(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Tesseract to detect text
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Confidence level of text detection
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
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
    st.image(image_no_text, caption='Generated Image', channels="BGR")
