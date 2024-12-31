import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from PIL import Image
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Load the model
model_dict = pickle.load(open('D:/project/model.pb', 'rb'))
model = model_dict['model']

# --- Text-to-Video Model Setup ---
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 42: 'space'
}

# Initialize Streamlit session state
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "final_output" not in st.session_state:
    st.session_state.final_output = ""

# Streamlit UI components
st.sidebar.title("Real-Time Sign Language Prediction with Text to video Generator")

start_button = st.sidebar.button("Start Webcam")
stop_button = st.sidebar.button("Stop Webcam and show text output")
clear_button = st.sidebar.button("Double click to Clear Output")
final_output_button = st.sidebar.button("Show Final Output")
submit_button = st.sidebar.button("generate text to video")


# Display final output in Streamlit
video_frame = st.empty()
predicted_text = st.empty()
st.info(f"Text Output: {st.session_state.final_output}")

# Clear output when the Clear button is clicked
if clear_button:
    st.session_state.final_output = ""

# Display final output when the Final Output button is clicked
if final_output_button:
    st.sidebar.success("Final Predicted Output:")
    st.sidebar.write(st.session_state.final_output)

# Function to update the text based on prediction
def append_to_final_output(character):
    if character == 'space':
        st.session_state.final_output += " "  # Append space for 'space' character
    else:
        st.session_state.final_output += character  # Append the predicted character

# Function to update the video feed in the Streamlit app
def update_video_feed(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    video_frame.image(image, channels="RGB", use_column_width=True)

# Function to run video capture and ASL prediction
def run():
    cap = cv2.VideoCapture(0)
    last_detected_character = None
    fixed_character = ""
    delayCounter = 0
    start_time = time.time()

    while st.session_state.run_camera:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw a rectangle and the predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                current_time = time.time()

                # Timer logic: Check if the predicted character is the same for more than 1 second
                if predicted_character == last_detected_character:
                    if (current_time - start_time) >= 1.0:  # Class fixed after 1 second
                        fixed_character = predicted_character
                        if delayCounter == 0:  # Add character once after it stabilizes for 1 second
                            append_to_final_output(fixed_character)
                            delayCounter = 1
                else:
                    # Reset the timer when a new character is detected
                    start_time = current_time
                    last_detected_character = predicted_character
                    delayCounter = 0  # Reset delay counter for a new character

        # Update Streamlit UI with the frame and current final output
        update_video_feed(frame)
        #st.info(f"Final Output: {st.session_state.final_output}")

    cap.release()
    cv2.destroyAllWindows()

# Text-to-video generation logic
if submit_button:
    st.session_state.video_prompt = st.session_state.final_output
    if st.session_state.video_prompt.strip():
        st.write(f"Using final output as prompt: {st.session_state.video_prompt}")
        video_duration_seconds = st.slider("Video Duration (seconds)", 1, 10, 4)

        with st.spinner("Generating video... This may take a while."):
            num_frames = video_duration_seconds * 10
            result = pipe(st.session_state.video_prompt, negative_prompt="", num_inference_steps=25, num_frames=num_frames)
            video_path = export_to_video(result.frames)

            st.video(video_path)
            st.success("Video generated successfully!")

            with open(video_path, "rb") as file:
                st.download_button("Download Video", file, "generated_video.mp4", "video/mp4")
    else:
        st.error("Final output is empty. Please generate text first.")


# Start/Stop Webcam Control
if start_button:
    st.session_state.run_camera = True
    run()

if stop_button:
    st.session_state.run_camera = False



# pip install mediapipe streamlit diffusers opencv-python torch

