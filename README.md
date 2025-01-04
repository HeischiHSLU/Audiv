# Audiv - DIGCRE | Livio & Hannes

Github from Audiv, our work consists of the pipeline of several machine learning processes. For you to use and/or try our Project you have to have an [Replicate](https://replicate.com/) Account with active api key. 

## Installation Audiv

1. Open Pytorch Environment (we used Jupyterhub https://gpuhub.labservices.ch)
2. Install everything from "PreRun.ipynb"
3. Add your Replicate API_token on line 32 inside "Audiv.py"
4. Open a Terminal and run "Audiv.py"
command: python3 Audiv.py
5. Open the gradio-Link generated in the console
6. Upload a Video
7. Choose between a screenshot
8. Enjoy your new Sound (this can take up to 10min if you are running it with standart Replicate)

## Usage

With Audiv, you can upload a video (e.g., from your newest project), and the AI will give you the option to choose between two screenshots from your video. After you’ve chosen one, the AI will generate a fitting and descriptive prompt for an audio track and pass it on to another AI, which will create the sound for you. In the end, you receive a audio file for your video.

## Process

### Ideation:

In Digital Ideation, it is common to have a demo of your newly created project. In our case, this is mostly done via a video. Unfortunately, in nearly every case, we didn’t have the time to produce a clean and polished video. This often leads to feedback about the video but not detailed feedback about our project. The most common feedback was that our videos lacked audio. We thought this module would be perfect to create an AI that generates fitting audio — and that’s exactly what we did.


### Functionality

Our Project can be roughly cut into 3 Pieces:  Video Cuting, Mood generation and Sound generation. For each of those 3 there will be a short section below talking about where our Project does these things inside our Python skript

**Video2Image**

For us to Process our Video easy and not having to have an complex Video analysation we use a Single Frame of a video instead of the whole Video. If u upload a Video on Gradio u will have the option to pick between the 33% Frame and the 66% Frame. We use this methode as it was easy and Quick to implement in Gradio.  
```python
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
```

**Image2Text**

Now that we have an image we use the an Replicate query from yorickvp/llava-13b to get Discription of the Mood of the image. To get a Discription we can use We worked out an Prompt that gives us a reliable Discription "Describe a fitting Background Music with less than 30 words. It must give an answer to the following 3 points: atmosphere, rhythm, melody"
```python
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
```
**Text2Audio**

Now the most resource-intensive part of our Pipline, the generation of the Audio itself. With the now generate Description of Mood Melody and Rythm of our Video we can use the "audio-ldm" form haohgeliu to generate the Audio. This is our finla Replicate Query and will output the wav. file wich we will then forward to the Gradio Interface
```python
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
```
