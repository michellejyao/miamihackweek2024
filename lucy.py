import cv2
import base64
import requests
import os
import speech_recognition as sr
import numpy as np
from gtts import gTTS

from playsound import playsound

from pathlib import Path
from openai import OpenAI

api_key = "sk-secret"
client = OpenAI(api_key=api_key)

cam = cv2.VideoCapture(0)
r = sr.Recognizer()

# reads an image file and encodes it to a base64 string, similar to the previous code.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# The analyzeVideo function captures an image from the webcam, saves it, and processes it.
# It sends the captured image (as a base64 string) along with a prompt to OpenAI's GPT-4 model for analysis.
# The response from the model is returned for further use.

def analyzeVideo(prompt):
  global ret, img
  ret, img = cam.read()

  os.remove("picture.png") #reset
  img_name = "picture.png".format(0)

  cv2.imwrite(img_name, img)
  image_name = "picture.png"
  
  playsound("processingsound.mp3")

  print("thinking...")


  # Getting the base64 string
  base64_image = encode_image(image_name)

  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }

  payload = {
      "model": "gpt-4-turbo",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt + ". Make any assumptions, keep responses to 2 sentences, and make sure you are descriptive to aid the visually impaired. Only refer the images only if need to."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 50
  }

  
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  content = response.json()["choices"][0]['message']['content']
  return(content)

# The listen_for_wake_word function listens for a specific wake word ("lucy").
# When detected, it calls the analyzeVideo function with the recognized speech text and converts the response to speech.
def listen_for_wake_word(source):
  
  while True:
      r.adjust_for_ambient_noise(source, duration=0.2)
      audio = r.listen(source)
      try:
          # The recognize_google method sends the audio data to Google's Speech Recognition service.
          # This service processes the audio and attempts to transcribe it into human-readable text.
          text = r.recognize_google(audio)
          print(text)
          if "lucy" in text.lower():
              print("hit")

              response_text = analyzeVideo(text)
              toVoice(response_text)
              print(response_text)

              
      except sr.UnknownValueError:
          pass

def toVoice(text):
  
  speech_file_path = Path(__file__).parent / "speech.mp3"
  response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=text
  )

  response.stream_to_file(speech_file_path)
  playsound("speech.mp3")  

with sr.Microphone() as source:
  listen_for_wake_word(source)
