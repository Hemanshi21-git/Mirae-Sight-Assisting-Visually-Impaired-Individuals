import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pytesseract
import pyttsx3
import torch
import cv2
import numpy as np

genai.configure(api_key="YOUR_GOOGLE_GEMINI_KEY")

# Initialize Google Generative AI
model = genai.GenerativeModel("gemini-1.5-pro")
st.set_page_config(page_title="MiraeSight", layout="centered", page_icon="🔮")

# Title and layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("MiraeSight🔮")
st.subheader(":blue[Mirae means future, focusing on forward-looking vision assistance]")

st.sidebar.image("Logo.jpg", width=250)
st.sidebar.markdown("""
**Features of MiraeSight 🌟:**
- **Real-Time Scene Analysis:** 🖼️ Instantly analyze and describe uploaded images for a better understanding of surroundings.
- **Object and Obstacle Detection:** 🚧 Detect objects and obstacles in real-time for safe navigation.
- **Personalized Assistance:** 💡 Provide tailored guidance and actionable insights based on detected objects and scene context.
- **Text-to-Speech Conversion:** 🔊 Seamlessly convert text into audio descriptions, enabling users to receive information audibly for better accessibility.
""")

st.sidebar.markdown(""" Built with ❤️ using Streamlit Using Google Gemini @MIRAESIGHT🔮""")


st.markdown("""**Upload an Image: 🖼️**""")
uploaded_file = st.file_uploader("Drag and drop or browse an image....", type=["jpg", "png", "jpeg","webp"])

# Convert image to bytes
def convert_image_to_bytes(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    return [{"mime_type": uploaded_file.type, "data": bytes_data}]

# Extract text from image
def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text.strip() or "No text found in the image."
    except Exception as e:
        return f"Error: {e}"

# Text-to-speech
def text_to_speech_pyttsx3(text): 
    try: 
        engine = pyttsx3.init()
        engine.say(text) 
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        raise RuntimeError(f"Failed to convert text to speech. Error: {e}")
# YOLO object detection
@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

yolo_model = load_yolo_model()


def generate_task_guidance(objects, image_content):
        """Generates personalized guidance based on detected objects and text."""

        prompt = (
            f"""Objects Detected: {', '.join(objects) if objects else 'None'}. "
            "Provide actionable insights or task-specific suggestions
            to help visually impaired people based on objects and image provided."
            "Be concise, clear, and practical."""
        )

        response = model.generate_content([prompt, image_content[0]])
        return response.text

# Buttons
bt1, bt2, bt3, bt4, bt5 = st.columns(5)
real_scene_button = bt1.button("Describe Scene 📸")
object_detection_button = bt2.button("Detect Objects 🔍")
assistance_button = bt3.button("Assist Tasks 🛠️")
text_button = bt4.button("Extract Text 📄")
tts_button = bt5.button("Text-to-Speech 🔊")
stop_audio_button = st.button("Stop Audio ⏹️")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if real_scene_button:
        st.spinner("Generating Description....")
        st.subheader("🌅 Scene Description::")
        image_content = convert_image_to_bytes(uploaded_file)
        scene_prompt = "Describe the scene in the image."
        response = model.generate_content([scene_prompt, image_content[0]])
        scene_description = response.text 
        st.write(f"Scene Description: {scene_description}") 
        # Convert scene description to speech using pyttsx3 
        text_to_speech_pyttsx3(scene_description) 
        st.write("Scene Description Text-to-Speech Conversion Completed.")
        

    if object_detection_button:
        st.spinner("Detecting objects....")
        st.subheader("🔍 Detected Objects:")
        # Convert image to OpenCV format
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Perform object detection
        results = yolo_model(image_bgr)
        results_df = results.pandas().xyxy[0]

        # Annotate the image
        for _, row in results_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Box thickness is 2
        
        # Adjust font properties for less bold labels
            font_scale = 0.5  # Reduce the font size
            font_thickness = 1  # Reduce the font thickness
            text_color = (255, 0, 0)  # Text color (blue)
        
        # Draw the label with adjusted properties
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(image_bgr, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1) 
            cv2.putText(image_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        annotated_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_image, caption="Detected Objects")#, use_column_width=True)

        st.write("Detection Details:")
        detected_objects=[]
        for _, row in results_df.iterrows():
            st.write(
                f"- Object: {row['name']}, Confidence: {row['confidence']:.2f}, "
                f"Bounding Box: ({row['xmin']:.0f}, {row['ymin']:.0f}), ({row['xmax']:.0f}, {row['ymax']:.0f})"
            )
            
    if assistance_button:
        st.spinner("Assiting....")
        st.subheader("🤝 Assitance:")
        detected_objects = [row['name'] for _, row in results_df.iterrows()] if 'results_df' in locals() else []
        # extracted_text = extract_text_from_image(uploaded_file)
        image_content = convert_image_to_bytes(uploaded_file)
        guidance = generate_task_guidance(detected_objects, image_content)
        st.write(f"Task Guidance: {guidance}")
        text_to_speech_pyttsx3(guidance)
        st.write("Assitance text-to-speech conversion completed.")
        
    if text_button:
        st.spinner("Exacting text....")
        st.subheader("📜 Exacted Text:")
        extracted_text = extract_text_from_image(uploaded_file)
        st.write(f"Extracted Text: {extracted_text}")
        
    if tts_button: 
        extracted_text = extract_text_from_image(uploaded_file)
        if extracted_text:
            text_to_speech_pyttsx3(extracted_text) 
            st.write("Text-to-Speech Conversion Completed.")
            
    if stop_audio_button:
     try:
        # Initialize TTS engine if not already initialized
        if "tts_engine" not in st.session_state:
            st.session_state.tts_engine = pyttsx3.init()
        
        # Stop the audio playback
        st.session_state.tts_engine.stop()
        st.success("Audio playback stopped.")
     except Exception as e:
        st.error(f"Failed to stop the audio. Error: {e}")