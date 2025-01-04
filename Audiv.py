import time
import os
import asyncio
import threading
from PIL import Image
import base64
import io
import requests
import json
import sys

try:
    import replicate
    print("replicate is installed.")
except ImportError:
    print("replicate is NOT installed.")

try:
    import cv2
    print("opencv-python is installed.")
    print("Version:", cv2.__version__)
except ImportError:
    print("opencv-python is NOT installed.")

try:
    import gradio as gr
    import numpy as np
    print("gradio is installed.")
except ImportError:
    print("gradio is NOT installed.")

client = replicate.Client(api_token="YOUR_API_TOKEN")

selectedImage = "https://replicate.delivery/pbxt/M1TEu9v1iHvdgEfj1A81dpEaYZ6RtK5PEIUSDmEhExOn5SIJ/GamePlay.jpg"
I2Tprompt = "Describe a fitting Background Music with less than 30 words. It must give an answer to the following 3 points: atmosphere, rhythm, melody"
T2APrompt = "-"
AudioOutput = None
base64_image = None
output_path_33 = "screenshot_33percent.jpg"
output_path_66 = "screenshot_66percent.jpg"

def process_video(video, radio):
    if (radio == "33%" or radio == "66%"):
        if(radio == "33%"):
            ImageToText(output_path_33, I2Tprompt)
        if(radio == "66%"):
            ImageToText(output_path_66, I2Tprompt)
    if (radio != "33%" or radio != "66%"):
        return capture_frames(video, output_path_33, output_path_66)
    return false

def capture_frames(video_path, output_path_33, output_path_66):
    global AudioOutput
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Cannot open the video file. Check the file path.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_33 = int(total_frames * 0.33)
    frame_66 = int(total_frames * 0.66)
    
    def save_frame(frame_number, output_path):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to capture the frame at {frame_number}.")
        cv2.imwrite(output_path, frame)
        print(f"Screenshot saved at {output_path}")
    
    save_frame(frame_33, output_path_33)
    save_frame(frame_66, output_path_66)
    
    cap.release()
    
    image33 = Image.open(output_path_33)
    image66 = Image.open(output_path_66)
    return [image33, image66, T2APrompt, AudioOutput]

def ImageToText(image_path,prompt):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    global T2APrompt
    output = client.run(
            "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
            input={
                "image": f"data:image/jpeg;base64,{base64_image}",
                "top_p": 1,
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.2,
            },)
    T2APrompt = ""
    for item in output:
        print(item, end="")
        T2APrompt += item
    TextToAudio(T2APrompt)
    return output

def TextToAudio(prompt):
    global AudioOutput
    print("Start Audio")
    AudioOutput = str(client.run(
    "haoheliu/audio-ldm:b61392adecdd660326fc9cfc5398182437dbe5e97b5decfb36e1a36de68b5b95",
    input={
        "text": T2APrompt,
        "duration": "5.0",
        "n_candidates": 3,
        "guidance_scale": 2.5
    }))
    print(AudioOutput)

def create_interface():
    interface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload a Video"),
        gr.Radio(choices=["33%", "66%"], label="Select Screenshot")
    ],
    outputs=[
        gr.Image(label="Screenshot at 33%"), 
        gr.Image(label="Screenshot at 66%"),
        gr.Textbox(label="Music Description"),  # Output: Music description
        gr.Audio(label="Generated Audio")  # Output: Audio generated from description
    ],
    live=True,
    flagging_mode="never"
)
    
    interface.launch(share=True)

create_interface()
print("\^v^/")
