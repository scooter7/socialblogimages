import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from detextify import Detextify

# Ensure that NLTK stopwords are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Set OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Instantiate Detextify
detextifier = Detextify()

# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if not word.lower() in stop_words]
    return ' '.join(filtered_text)

# Function to generate image using OpenAI's DALL-E
def generate_image(prompt, width, height):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=f"{width}x{height}"
    )
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to remove text from the image using Detextify
def remove_text_from_image(image):
    result_image = detextifier.remove_text(image)
    return result_image

# Streamlit application
st.title('Text to Image Generator')

# Text input for dimensions
width = st.number_input('Width', min_value=256, max_value=1024, value=512)
height = st.number_input('Height', min_value=256, max_value=1024, value=512)

# Text input for user content
user_input = st.text_area("Enter your text here:")

# Button to process text and generate image
if st.button('Generate Image'):
    # Extract keywords from the user input
    keywords = extract_keywords(user_input)

    # Generate image
    image = generate_image(keywords, width, height)

    # Remove text from image
    image_without_text = remove_text_from_image(image)

    # Display the image
    st.image(image_without_text, caption='Generated Image')
