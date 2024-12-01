import streamlit as st
import google.generativeai as genai

genai.configure(api_key="Your_key")

# Initialize Google Generative AI
model = genai.GenerativeModel("gemini-1.5-pro")

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

st.title("AI-Powered Scene Understanding")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # image = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Scene Description")
    scene_description = generate_scene_description(uploaded_image)
    st.write(scene_description)