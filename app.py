import streamlit as st
from PIL import Image
import cv2
import os
import time
import torch
from ultralytics import YOLO
from collections import Counter
import openai


openai.api_key = "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"

st.set_page_config(layout="wide")

cfg_model_path = 'models/best 908.pt'
model = None
confidence = 0.45


def image_input():
    with st.container():
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_file, caption="Selected Image")
            with col2:
                img, result = infer_image(Image.open(img_file))
                process_results(result)


def video_input():
    with st.container():
        vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi', 'mkv'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

            cap = cv2.VideoCapture(vid_file)
            process_video_stream(cap)


def live_input():
    cap = cv2.VideoCapture(0)  # Open the default camera
    process_video_stream(cap)


def process_video_stream(cap):
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

    fps = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")

    st.markdown("---")
    output = st.empty()
    prev_time = 0
    curr_time = 0
    stop = st.button("Stop")
    identified_vegetables = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Can't read frame, stream ended? Exiting ....")
            break
        frame = cv2.resize(frame, (width, height))
        output_img, result = infer_image(frame)
        output.image(output_img)
        counter = quantity_estimate(result)
        identified_vegetables.update([result[0].names[k] for k in counter.keys()])
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")
        if stop:
            cap.release()
            break

    cap.release()

    # Process results after video/live feed
    if identified_vegetables:
        process_results(identified_vegetables)


def process_results(identified_items):
    if isinstance(identified_items, set):
        identified_vegetables = list(identified_items)
    else:
        counter = quantity_estimate(identified_items)
        identified_vegetables = [identified_items[0].names[k] for k in counter.keys()]

    st.write("Identified Vegetables:")
    for veg in identified_vegetables:
        st.write(f"- {veg}")

    st.markdown("### What cuisine would you like?")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Indian"):
            st.session_state["cuisine"] = "Indian"
    with col2:
        if st.button("Italian"):
            st.session_state["cuisine"] = "Italian"
    with col3:
        if st.button("French"):
            st.session_state["cuisine"] = "French"
    with col4:
        if st.button("Mexican"):
            st.session_state["cuisine"] = "Mexican"
    with col5:
        if st.button("Thai"):
            st.session_state["cuisine"] = "Thai"

    if "cuisine" in st.session_state:
        generate_recipes_for_cuisine(identified_vegetables, st.session_state["cuisine"])


def generate_recipes_for_cuisine(vegetables, cuisine):
    if "recipes" not in st.session_state or st.session_state["cuisine"] != cuisine:
        st.session_state["recipes"] = get_recipe_names(vegetables, cuisine)
        st.session_state["cuisine"] = cuisine

    recipes = st.session_state["recipes"]
    if recipes:
        st.markdown(f"### {cuisine} Recipes")
        for recipe in recipes:
            if st.button(recipe, key=recipe):
                st.session_state["selected_recipe"] = recipe
                detailed_recipe = get_detailed_recipe(recipe, cuisine)
                st.session_state["detailed_recipe"] = detailed_recipe
                #st.experimental_rerun()

    if "selected_recipe" in st.session_state and "detailed_recipe" in st.session_state:
        st.markdown(f"### {st.session_state['selected_recipe']} - Detailed Recipe")
        st.markdown(st.session_state["detailed_recipe"])


def get_recipe_names(vegetables, cuisine):
    """Use ChatGPT to generate recipe names based on the vegetables and cuisine."""
    prompt = (
        f"I have the following vegetables: {', '.join(vegetables)}. "
        f"Can you suggest at least 5 recipe names that can be made in {cuisine} cuisine? "
        "Only return the names of the recipes."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        recipes = response['choices'][0]['message']['content'].split("\n")
        recipes = [r.strip("- ").strip() for r in recipes if r.strip()]  # Clean and format the output
        return recipes[:5]  # Ensure at least 5 recipes are returned
    except Exception as e:
        st.error(f"Error fetching recipe names: {e}")
        return []


def get_detailed_recipe(recipe_name, cuisine):
    """Use ChatGPT to generate a detailed recipe for the selected dish."""
    prompt = (
        f"Provide a detailed recipe for the dish '{recipe_name}', which is part of {cuisine} cuisine. "
        "Include the ingredients and step-by-step instructions."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error fetching detailed recipe: {e}")
        return "Unable to fetch the recipe. Please try again."


def infer_image(frame, size=None):
    results = model.predict(source=frame, show=False, conf=confidence, save=False)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
    return im, results


def quantity_estimate(result):
    counter = Counter(result[0].boxes.cls.numpy().astype(int))
    return counter


@st.cache_resource
def load_model(cfg_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ = YOLO(cfg_model_path)
    model_.to(device)
    print("Model loaded on", device)
    return model_


def main():
    global model, confidence, cfg_model_path

    # Apply custom styles
    st.markdown(
    """
    <style>
    /* Background styling */
    body {
        background-image: url('images/background.jpg'); /* Adjust path as needed */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff; /* Default text color */
    }

    /* Import custom font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Style for file uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
        border-radius: 10px;
        padding: 15px;
        color: #1e3c72; /* Matching text color */
        border: 1px solid #90caf9; /* Light border */
    }

    /* Style for buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #6fb1fc, #1e90ff); /* Gradient matching the theme */
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    /* Button hover effect */
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(30, 144, 255, 0.6); /* Light shadow effect */
    }

    /* Style for images */
    img {
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); /* Subtle shadow for images */
    }

    /* Center headings with a shadow effect */
    h1, h2, h3 {
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Style for container elements */
    .stContainer {
        background: rgba(255, 255, 255, 0.8); /* Semi-transparent background */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Soft shadow */
    }

    /* Style for recipe text */
    .recipe-text {
        font-size: 18px;
        line-height: 1.6;
        color: #1e3c72;
        background: rgba(255, 255, 255, 0.8); /* Semi-transparent background for recipes */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    </style>
    """,
        unsafe_allow_html=True
    )

    st.title("Smart AI Recipe Genius")

    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please add to the model folder.", icon="⚠️")
    else:
        model = load_model(cfg_model_path)

        confidence = 0.45

        model.classes = list(model.names.keys())

        st.markdown("### Select Input Type:")
        input_option = st.radio("Choose the input type:", ['🖼️ Image', '📹 Video', '📷 Live'], horizontal=True)

        st.markdown("---")  # Add a horizontal separator

        # Render the appropriate input method
        if input_option == '🖼️ Image':
            st.markdown("### Upload an Image")
            image_input()
        elif input_option == '📹 Video':
            st.markdown("### Upload a Video")
            video_input()
        else:
            st.markdown("### Use Live Camera")
            live_input()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
