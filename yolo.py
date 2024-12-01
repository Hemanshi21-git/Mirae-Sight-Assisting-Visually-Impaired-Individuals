import google.generativeai as genai
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s' for small model

st.title("Object and Obstacle Detection for Safe Navigation")
st.write("Upload an image to detect objects and obstacles.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(image_bgr)

    # Convert results to pandas DataFrame for easy processing
    results_df = results.pandas().xyxy[0]  # Bounding box format: xmin, ymin, xmax, ymax

    # Annotate the image with detected objects
    for _, row in results_df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        
        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Box thickness is 2
        
        # Adjust font properties for less bold labels
        font_scale = 0.5  # Reduce the font size
        font_thickness = 1  # Reduce the font thickness
        text_color = (255, 0, 0)  # Text color (blue)
        
        # Draw the label with adjusted properties
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(image_bgr, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)  # Text background
        cv2.putText(image_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Convert BGR to RGB for Streamlit display
    annotated_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Display results
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Extract details of detected objects
    st.write("Detection Details:")
    detected_objects = []
    for _, row in results_df.iterrows():
        detected_objects.append(row['name'])
        st.write(
            f"- Object: {row['name']}, Confidence: {row['confidence']:.2f}, "
            f"Bounding Box: ({row['xmin']:.0f}, {row['ymin']:.0f}), ({row['xmax']:.0f}, {row['ymax']:.0f})"
        )


    # Generate task guidance based on detected objects and text
    def generate_task_guidance(objects):
        """Generates personalized guidance based on detected objects and text."""
        genai.configure(api_key='AIzaSyDHdZMngWXmsJGb3uO2-ZWVER4HlM6MTsY') # Replace with your actual Gemini API key
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        prompt = (
            f"Objects Detected: {', '.join(objects) if objects else 'None'}. "
            "Provide actionable insights or task-specific suggestions to help visually impaired people. "
            "Be concise, clear, and practical."
        )

        response = model.generate_content(prompt)
        return response.text

    guidance = generate_task_guidance(detected_objects)
    st.write("Personalized Task Guidance:")
    st.write(guidance)

