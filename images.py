import streamlit as st
import openai_secret_manager
import requests
from PIL import Image
import numpy as np
import easyocr
import cv2

assert 'openai' in openai_secret_manager.get_services()
openai_api_key = openai_secret_manager.get_secret("openai")["api_key"]

st.title("Image Generation & Text Removal App")

st.markdown("## Step 1: Generate Image from Text")
text_prompt = st.text_area("Enter a text prompt to generate an image:", height=150)
width = st.selectbox("Select image width:", [1024, 1792], index=0)
height = st.selectbox("Select image height:", [1024, 1792], index=0)

if st.button("Generate Image"):
    # Adjust width and height for aspect ratio
    if width < height:
        size = "1024x1792"
    elif width > height:
        size = "1792x1024"
    else:
        size = "1024x1024"

    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": text_prompt,
            "n": 1,
            "size": size,
        },
    )
    response.raise_for_status()
    data = response.json()
    image_url = data['data'][0]['url']
    image_response = requests.get(image_url)
    image_response.raise_for_status()
    
    with open('generated_image.png', 'wb') as f:
        f.write(image_response.content)

    st.image('generated_image.png', caption='Generated Image', use_column_width=True)

st.markdown("## Step 2: Remove Text from an Image")
uploaded_image = st.file_uploader("Or upload your image here to remove text:", type=['png', 'jpg', 'jpeg'])

@st.cache(allow_output_mutation=True)
def load_model(): 
    reader = easyocr.Reader(['en'], model_storage_directory='.')
    return reader 

reader = load_model()  # Load model

def remove_text_from_image(image, result):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    mask = np.zeros(image_cv.shape[:2], np.uint8)
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        top_left = tuple(map(int, tl))
        bottom_right = tuple(map(int, br))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    inpainted_image = cv2.inpaint(image_cv, mask, 7, cv2.INPAINT_TELEA)
    inpainted_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    return inpainted_image

if uploaded_image is not None:
    input_image = Image.open(uploaded_image)
    st.image(input_image, caption='Uploaded Image')

    if st.button("Remove Text from Uploaded Image"):
        with st.spinner("Removing text..."):
            result = reader.readtext(np.array(input_image))
            inpainted_image = remove_text_from_image(input_image, result)
            st.image(inpainted_image, caption='Image with Text Removed')
        st.success("Text has been removed!")
else:
    st.write("Upload an Image to remove text.")

st.caption("Made with ❤️ by @1littlecoder")
