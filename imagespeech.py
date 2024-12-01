import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pytesseract
import pyttsx3

genai.configure(api_key="AIzaSyCiNjOZjv1BWabI5QcLit6zVaoExABxbDY")

# Initialize Google Generative AI
model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# New function to convert image to bytes
def convert_image_to_bytes(uploaded_file):
    bytes_data = uploaded_file.getvalue()

    image_parts = [
        {
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }
    ]

    return image_parts

# Function to generate scene description
def generate_scene_description(image):
    image_content = convert_image_to_bytes(image)

    scene_prompt = "Describe the scene in the image."
    
    # Invoke the model
    response = model.generate_content([scene_prompt, image_content[0]])
    
    return response.text

# Function to extract text from the image
def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)

        # pytesseract to extract text
        extracted_text = pytesseract.image_to_string(img)

        if not extracted_text.strip():
            return "No text found in the image."
        
        return extracted_text

    except Exception as e:
        raise ValueError(f"Failed to extract text. Error: {e}")

# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

st.title("AI-Powered Scene Understanding")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # image = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Scene Description")
    scene_description = generate_scene_description(uploaded_image)
    st.write(scene_description)
    
    # Convert the scene description to speech
    text_to_speech(scene_description)

    # Optional: Extract text from image (if required)
    extracted_text = extract_text_from_image(uploaded_image)
    st.subheader("Extracted Text from Image")
    st.write(extracted_text)
    
    # Convert the extracted text to speech (if any)
    if extracted_text.strip():
        text_to_speech(extracted_text)
